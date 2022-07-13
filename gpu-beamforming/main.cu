#include <iostream>
#include <cuComplex.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <cusolverDn.h>
#include "device.cu"

const double SPEED_OF_LIGHT = 299'792'458.0f;
cudaError err = cudaSuccess;

const double AZIMUTH = (40.0f / 360.0f) * 2 * M_PI;
const double ELEVATION = (40.0f / 360.0f) * 2 * M_PI;
const double INCIDENT_FREQUENCY = 642'000'000.0f;
const int M = 20; // number of elements in the array

struct Sensor {
private:
    double3 pos;
    int index;
    cuDoubleComplex *output;
public:
    Sensor(double x,
           double y,
           double z,
           int index) : index(index) {
        pos.x = x;
        pos.y = y;
        pos.z = z;
    };

    Sensor() : pos(), index() {};

    friend std::ostream &operator<<(std::ostream &os,
                                    const Sensor obj) {
        os << "Element " << obj.index << " is at (x, y, z): (" << obj.get_pos().x << ", " << obj.get_pos().y << ", "
           << obj.get_pos().z << ")";
        return os;
    }

    double3 get_pos() const {
        return this->pos;
    }

    void set_pos_and_index(double pos_x,
                           double pos_y,
                           double pos_z,
                           int idx) {
        this->pos.x = pos_x;
        this->pos.y = pos_y;
        this->pos.z = pos_z;
        this->index = idx;
    }

    double calculate_time_delay(double3 pos_target) const {
        double dist = sqrt(pow(pos.x - pos_target.x, 2) + pow(pos.y - pos_target.y, 2) + pow(pos.z - pos_target.z, 2));
        return dist / SPEED_OF_LIGHT;
    }
};

struct Array {
    Sensor *antenna_elements;
    int2 ref_element;
    double3 ref_point;
    cuDoubleComplex *steering_vector, *steering_vector_c;
    int2 antenna_dim;

    ~Array() {
        free(antenna_elements);
        free(steering_vector);
        free(steering_vector_c);
    }

    /**@Description
     * Initialise the array with a specified reference element and point in space, with each sensor equidistantly
     * positioned (by _distance_ length units) in a planar array of size _rows_ X _cols_. The steering vector (matrix)
     * of the array is calculated based on the predefined macros INCIDENT_FREQUENCY, AZIMUTH and ELEVATION as well as
     * the distances passed to this constructor.
     *
     * @param ref_element the index of the array element considered to be the reference
     * @param distance the distance by which the array elements are separated from each other
     * @param ref_point the point where ref_element is located
     * @param d_x distance between elements in the x-direction
     * @param d_y distance between elements in the y-direction
     * @param rows the number of rows of elements in the array aperture
     * @param cols the number of columns of elements in the array aperture
     */
    Array(int2 ref_element,
          double3 ref_point,
          double d_x = 0.07f,
          double d_y = 0.07f,
          int rows = 4,
          int cols = 6)
            : ref_element(ref_element), ref_point(ref_point) {
        antenna_elements = (Sensor *) malloc(rows * cols * sizeof(Sensor));
        steering_vector = (cuDoubleComplex *) malloc(rows * cols * sizeof(cuDoubleComplex));
        steering_vector_c = (cuDoubleComplex *) malloc(rows * cols * sizeof(cuDoubleComplex));
        antenna_dim = {rows, cols};
        double distance_x, distance_y, electrical_angle, distance; // distance to ref_element
        int offset;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                offset = i + rows * j;
                // this case is needed to match the array setup, as one column of the array was not functional at the time of gathering data
                if (j >= 3) {
                    distance_x = (j - ref_element.x + 1) * d_x;
                } else {
                    distance_x = (j - ref_element.x) * d_x;
                }
                distance_y = (i - ref_element.y) * d_y;
                distance = sqrt(pow(distance_x, 2) + pow(distance_y, 2));
                electrical_angle = 2 * M_PI * INCIDENT_FREQUENCY / SPEED_OF_LIGHT * sin(AZIMUTH) * (distance_x * cos(ELEVATION) + distance_y * sin(ELEVATION));
                steering_vector[offset] = make_cuDoubleComplex(cos(electrical_angle), sin(electrical_angle));
                steering_vector_c[offset] = make_cuDoubleComplex(cos(electrical_angle), -sin(electrical_angle));
                antenna_elements[offset].set_pos_and_index(ref_point.x + distance_x, ref_point.y + distance_y,
                                                           ref_point.z,
                                                           offset);
            }
        }
    };

    Array() : ref_element({0, 0}), ref_point({0, 0, 0}) {
        double d_x = 0.07f, d_y = 0.07f;
        double distance = sqrt(pow(d_x, 2) + pow(d_y, 2));
        for (int i = 0; i < M; i++) {
            antenna_elements[i].set_pos_and_index(ref_point.x + distance * (i - ref_element.x), ref_point.y,
                                                  ref_point.z,
                                                  i); // If no dimensions are specified, use ULA geometry
        }
    };

    Sensor get_element_from_matrix_indices(int row,
                                           int col) const {
        return antenna_elements[row + col * antenna_dim.x];
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const Array &obj) {
        for (int i = 0; i < M; i++) {
            os << obj.antenna_elements[i] << std::endl << "Steering coefficient: " << cuCreal(obj.steering_vector[i])
               << " + " << cuCimag(obj.steering_vector[i]) << "i" << std::endl << std::endl;
        }
        return os;
    }
};

/** @Description
 * Calculates the optimal set of weights for the array elements from the direction vector
 *
 * @param steering_vector The direction vector of the desired signal (allocated in GPU memory)
 * @param size A variable for storing the calculated norm of the steering vector (allocated in CPU memory)
 * @param weights The array one wants to hold the calculated weights (allocated in GPU memory)
 * @param cublas_handle A handler for cublas operations, required for cublas scale and dot operations
 */
void calc_bartlett_weights_from_direction(const cuDoubleComplex *steering_vector,
                                          cuDoubleComplex &size,
                                          cuDoubleComplex *weights,
                                          cublasHandle_t &cublas_handle) {
    // Optimal weights in conventional beamforming is a(theta) / sqrt(a(theta)^Ha(theta))
    cudaMemcpy(weights, steering_vector, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    cublasZdotc_v2(cublas_handle, M, steering_vector, 1, steering_vector, 1, &size);
    size = {1 / sqrt(size.x), 0};
    cublasZscal_v2(cublas_handle, M, &size, weights, 1);
}

/** @Description
 * Function to clean up the shared memory and close input file
 *
 * @param weights
 * @param array_output
 * @param mul_result
 * @param array
 * @param file
 */
void
execute_cleanup(cuDoubleComplex *weights,
                Array *array,
                std::ifstream &file) {
    cudaFree(&weights);
    delete array;
    file.close();
}

/** @Description
 * This function wraps around the execution of a conventional beamformer, facilitating the calculation of weights and
 * stepping through the scenario
 *
 * @param array The object holding the array aperture geometry and steering matrix (vector in case of ULA)
 * @param weights The array pointer to where the calculated weights are to be stored
 * @param N The total number of time steps in the data
 * @param power_output The array pointer to where the beamformer output is to be stored
 * @param data The array pointer to where the unprocessed data is stored
 */
void execute_bartlett_beamforming(Array *array,
                                  cuDoubleComplex *weights,
                                  int N,
                                  double *power_output,
                                  cuDoubleComplex *data) {
    cuDoubleComplex res = {0, 0}, *array_output_d, *steering_vector_d;
    cublasHandle_t cublas_handle;
    std::chrono::steady_clock::time_point start, end;
    int data_index = 0, time_index;

    cublasCreate_v2(&cublas_handle);
    cudaMalloc(&array_output_d, M * sizeof(cuDoubleComplex));
    cudaMalloc(&steering_vector_d, M * sizeof(cuDoubleComplex));
    cudaMalloc((void **) &time_index, sizeof(int));
    cudaMemcpy(&data_index, &time_index, sizeof(int), cudaMemcpyHostToDevice);
    cublasSetVector(M, sizeof(cuDoubleComplex), array->steering_vector, 1, steering_vector_d, 1);


    // Step 2: calculate weights using the steering vector
    calc_bartlett_weights_from_direction(steering_vector_d, res, weights, cublas_handle);

    start = std::chrono::steady_clock::now();
    for (int t = 0; t < N; t++) {
        // Step 3: Get array output index corresponding to the first sensor at that time instance
        // by multiplying the time step by the number of elements
        data_index = t * M;

        // Step 4: Calculate response by multiplying the array output with the calculated weights
        cublasZdotc_v2(cublas_handle, M, weights, 1, &array_output_d[data_index], 1, &res);

        // Step 5: Calculate the Power output
        get_power<<<1, 1>>>(&res, power_output, time_index);

    }
    end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    cudaFree(&array_output_d);
    cudaFree(&steering_vector_d);
    cudaFree(&time_index);
    cublasDestroy_v2(cublas_handle);
    std::cout << "Processing took " << duration << " µs" << std::endl;
}

/** @Description
 * Calculates the adaptive weights for the MVDR beamformer using the Moore-Penrose pseudo inverse
 *
 * @param array_output An array pointer holding the current time step's array output (on GPU)
 * @param intermediate_result An array pointer to a temporary storage array of size M
 * @param weights An array pointer to where the computed weights are to be stored
 * @param steering_vector An array pointer to where the steering vector is stored on the GPU
 * @param steering_vector_c An array pointer to where the complex conjugate of the steering vector is stored on the GPU
 * @param cublas_handle A reference to a cublas handler instance (required for certain cuBLAS operations)
 * @param cusolver_dn_handle A reference to a cuSOLVER handler instance (required for SVD procedure)
 * @param scale A reference to a scaling factor used in certain cuBLAS operations
 * @param cross_spectral_matrix An array pointer to where the Cross Spectral Matrix (CSM) is to be stored during processing on the GPU
 * @param rec_sing_vals An array pointer to where the reciprocals of singular values resulting from the SVD procedure are to be stored
 * @param inverse_cross_spectral_matrix An array pointer to where the pseudo inverse of the CSM is to be stored
 * @param csm_scale A reference to another scaling factor used to negate the effect of memory that should not be taken into consideration in some operations
 * @param sing_vals An array pointer to where the singular values found by SVD
 * @param ls_vec An array pointer to where the left singular vector matrix is to be stored resulting from the SVD
 * @param rs_vec An array pointer to where the right singular vector matrix is to be stored resulting from the SVD
 * @param lwork A reference to the size of the working space of the SVD subprocess
 * @param d_work An array pointer to where the working space starts
 * @param d_rwork An array pointer to where real numbers resulting from intermediate steps of the SVD is stored
 * @param dev_info An error variable, pointing out which (if any) parameter in SVD is erroneous
 */
void calc_mvdr_weights_from_direction(cuDoubleComplex *array_output,
                                      cuDoubleComplex *intermediate_result,
                                      cuDoubleComplex *weights,
                                      cuDoubleComplex *steering_vector,
                                      cuDoubleComplex *steering_vector_c,
                                      cublasHandle_t &cublas_handle,
                                      cusolverDnHandle_t &cusolver_dn_handle,
                                      cuDoubleComplex &scale,
                                      cuDoubleComplex *cross_spectral_matrix,
                                      cuDoubleComplex *sample_covariance_matrix,
                                      cuDoubleComplex *rec_sing_vals,
                                      cuDoubleComplex *inverse_cross_spectral_matrix,
                                      cuDoubleComplex &csm_scale,
                                      double *sing_vals,
                                      cuDoubleComplex *ls_vec,
                                      cuDoubleComplex *rs_vec,
                                      int &lwork,
                                      cuDoubleComplex *d_work,
                                      double *d_rwork,
                                      int *dev_info,
                                      const int &batch_size,
                                      const int &time_step,
                                      const cuDoubleComplex &scm_scale) {
    // Estimate Cross Spectral Matrix (CSM) from array output (array_output * array_output^H)
    // The CSM is Hermitian, i.e. CSM = CSM^H
    cublasZgerc_v2(cublas_handle, M, M, &scale, array_output, 1, array_output, 1, cross_spectral_matrix, M);

    if (time_step % batch_size == 0) {
        // Scale down the accumulated size of the sample covariance matrix
        cublasZscal_v2(cublas_handle, M * M, &scm_scale, sample_covariance_matrix, 1);

        /** Calculate the Moore-Penrose pseudo inverse of the CSM */
        // Compute SVD (Singular Value Decomposition of CSM
        cusolverDnZgesvd(cusolver_dn_handle, 'A', 'A', M, M, sample_covariance_matrix, M, sing_vals, ls_vec, M, rs_vec, M,
                         d_work, lwork, d_rwork, dev_info);

        // overwrite previous batch sample covariance matrix with current batch's first cross spectral matrix
        cudaMemcpyAsync(sample_covariance_matrix, cross_spectral_matrix, M * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

        // Invert the singular values and place them along the diagonal of an M x M matrix
        invert<<<1, M * M>>>(M, sing_vals, rec_sing_vals);

        // Multiply the left- and right singular vectors together with the reciprocated singular values
        // to form the Moore-Penrose pseudo-inverse
        cublasZgemm_v2(cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N, M, M, M, &scale, rs_vec, M, rec_sing_vals, M, &csm_scale,
                       inverse_cross_spectral_matrix, M);
        cublasZgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_C, M, M, M, &scale, inverse_cross_spectral_matrix, M, ls_vec,
                       M, &csm_scale, inverse_cross_spectral_matrix, M);

        /** Calculate the weights as: CSM⁻¹ * steering_vector / (steering vector^H * CSM⁻¹ * steering_vector) */

        // CSM⁻¹ * steering_vector (numerator [vector])
        cublasZgemv_v2(cublas_handle, CUBLAS_OP_N, M, M, &scale, inverse_cross_spectral_matrix, M, steering_vector, 1,
                       &csm_scale, weights, 1);

        // steering_vector^H * CSM⁻¹ * steering_vector = (CSM⁻¹^T * steering_vector^*) * steering_vector (denominator [scalar])
        // (CSM⁻¹)^T * conj(steering_vector) = l
        cublasZgemv_v2(cublas_handle, CUBLAS_OP_T, M, M, &scale, inverse_cross_spectral_matrix, M, steering_vector_c, 1,
                       &csm_scale, intermediate_result, 1);
        // csm_scale = l * steering_vector
        cublasZdotu_v2(cublas_handle, M, intermediate_result, 1, steering_vector, 1, &csm_scale);

        // take reciprocal of the calculated denominator and multiply with the weight vector
        csm_scale = cuCdiv({1, 0}, csm_scale);
        cublasZscal_v2(cublas_handle, M, &csm_scale, weights, 1);

    } else {
        cublasZaxpy_v2(cublas_handle, M * M, &scale, cross_spectral_matrix, 1, sample_covariance_matrix, 1);
    }
}

/**
 * @Description
 * This function wraps around the execution of an MVDR beamformer, facilitating the calculation of weights and stepping through the scenario
 *
 * @param array The object holding the array aperture geometry and steering matrix (vector in case of ULA)
 * @param weights The array pointer to where the calculated weights are to be stored
 * @param N The total number of time steps in the data
 * @param power_output The array pointer to where the beamformer output is to be stored
 * @param data The array pointer to where the unprocessed data is stored
 */
void execute_mvdr_beamforming(Array *array,
                              cuDoubleComplex *weights,
                              const int &N,
                              double *power_output,
                              cuDoubleComplex *data,
                              const int &batch_size) {
    cuDoubleComplex *res, *array_out_d, *CSM_d, *U_d, *V_d, *d_work, *inv_CSM_d, *intermediate_result,
            *rec_sing_vals_d, *steering_vector_d, *steering_vector_c_d, scale = {1, 0}, csm_scale = {0, 0},
            *sample_covariance_matrix_d, *scm_init, scm_scale = {1.0f / (double) batch_size, 0};
    int lwork = 0, *dev_info = nullptr, *time_index_d;
    double *d_rwork, *S_d;
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_dn_handle;
    cublasCreate_v2(&cublas_handle);
    cusolverDnCreate(&cusolver_dn_handle);
    cudaMalloc((void **) &res, sizeof(cuDoubleComplex));
    cudaMalloc((void **) &array_out_d, M * sizeof(cuDoubleComplex));
    cudaMalloc((void **) &intermediate_result, M * sizeof(cuDoubleComplex));
    cudaMalloc((void **) &steering_vector_d, M * sizeof(cuDoubleComplex));
    cudaMalloc((void **) &steering_vector_c_d, M * sizeof(cuDoubleComplex));
    cudaMalloc((void **) &dev_info, sizeof(int));
    cudaMalloc((void **) &time_index_d, sizeof(int));
    cudaMalloc((void **) &S_d, M * M * sizeof(cuDoubleComplex));
    cudaMalloc((void **) &U_d, M * M * sizeof(cuDoubleComplex));
    cudaMalloc((void **) &V_d, M * M * sizeof(cuDoubleComplex));
    cudaMalloc((void **) &CSM_d, M * M * sizeof(cuDoubleComplex));
    cudaMalloc((void **) &sample_covariance_matrix_d, M * M * sizeof(cuDoubleComplex));
    cudaMalloc((void **) &inv_CSM_d, M * M * sizeof(cuDoubleComplex));
    cudaMalloc((void **) &rec_sing_vals_d, M * M * sizeof(cuDoubleComplex));

    scm_init = (cuDoubleComplex*) calloc(M * M, sizeof(cuDoubleComplex));

    cudaMemcpy(time_index_d, &lwork, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sample_covariance_matrix_d, scm_init, M * M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cusolverDnZgesvd_bufferSize(cusolver_dn_handle, M, M, &lwork);
    cudaMalloc((void **) &d_work, lwork * sizeof(double));
    cublasSetVector(M, sizeof(cuDoubleComplex), array->steering_vector, 1, steering_vector_d, 1);
    cublasSetVector(M, sizeof(cuDoubleComplex), array->steering_vector_c, 1, steering_vector_c_d, 1);


    std::chrono::steady_clock::time_point start, end;

    /**
     * A consideration: Maybe try to CPU parallelise over time?
     * The CPU core assigned to run this program is maxed out in workload, which is affecting the performance,
     * one run of the program takes 1h, 20mins
     */
    start = std::chrono::steady_clock::now();
    for (int t = 0; t < N; t += 20) {
        // Set array output vector: is it necessary? Could I not just propagate data[t * M] to weight determination?
        cublasSetMatrix(M, 1, sizeof(cuDoubleComplex), data + t * M, M, array_out_d, M);

        // Calculate weights with estimated Cross Spectral Matrix, weights fulfill requirement a^H(theta, phi) * w = 1.
        calc_mvdr_weights_from_direction(array_out_d, intermediate_result, weights, steering_vector_d,
                                         steering_vector_c_d, cublas_handle, cusolver_dn_handle, scale, CSM_d,
                                         sample_covariance_matrix_d, rec_sing_vals_d, inv_CSM_d, csm_scale, S_d,
                                         U_d, V_d, lwork, d_work, d_rwork, dev_info, batch_size, t, scm_scale);

        // y(t) = w^Hx(t) = cublasZdotc(weights, array_out_d), dotc indicates the conjugate of
        // the first argument vector is dot-multiplied with the second argument, and a row
        // vector multiplied by a column vector is simply the dot product between them
        cublasZdotc_v2(cublas_handle, M, weights, 1, array_out_d, 1, res);

        // P_BF = |y(t)|² = y(t)*y(t)_c
        get_power<<<1, 1>>>(res, power_output, *time_index_d);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Processing took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " µs" << std::endl;
    cudaFree(&array_out_d);
    cudaFree(&steering_vector_d);
    cudaFree(&steering_vector_c_d);
    cudaFree(&CSM_d);
    cudaFree(&d_work);
    cudaFree(&d_rwork);
    cudaFree(&S_d);
    cudaFree(&U_d);
    cudaFree(&V_d);
    cudaFree(&dev_info);
    cudaFree(&rec_sing_vals_d);
    cublasDestroy_v2(cublas_handle);
    cusolverDnDestroy(cusolver_dn_handle);
}

int main() {
    // declare variables
    int num_time_steps = 4194304, batch_size = 10000;
    double *power_output, data_tmp;
    Array *array;
    cuDoubleComplex *weights;

    // allocate shared memory for variables
    err = cudaMalloc(&weights, M * sizeof(cuDoubleComplex));
    err = cudaMalloc(&power_output, num_time_steps * sizeof(double));

    // Step 1: calculate steering vector (matrix) for array
    array = new Array({0, 0}, {0, 0, 0});

    // Reading data from disk into contiguous block and calculating array output
    std::ifstream in("data_files/2022-01-11.txt");
    cuDoubleComplex *data, *data_d;
    data = (cuDoubleComplex *) malloc(num_time_steps * M * sizeof(cuDoubleComplex));
    cudaMalloc((void **) &data_d, M * num_time_steps * sizeof(cuDoubleComplex));
    for (int t = 0; t < num_time_steps; t++) {
        for (int m = 0; m < M; m++) {
            in >> data_tmp;
            data[t * M + m] = cuCmul(make_cuDoubleComplex(data_tmp, 0), array->steering_vector[m]);
        }
    }

    cudaMemcpy(data_d, data, M * num_time_steps * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

//    execute_bartlett_beamforming(array, weights, num_time_steps, power_output, data_d);
    execute_mvdr_beamforming(array, weights, num_time_steps, power_output, data_d, batch_size);

    execute_cleanup(weights, array, in);
    cudaFree(&power_output);
    cudaFree(&data_d);
    return 0;
}
