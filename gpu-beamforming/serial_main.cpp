//
// Created by u080334 on 5/31/22.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <ctime>

const int M = 20;
const double SPEED_OF_LIGHT = 299'792'458.0f;

const double INCIDENT_ANGLE = (40.0f / 360.0f) * 2 * M_PI;
const double INCIDENT_FREQUENCY = 642'000'000.0f;

struct double3 {
    double x, y, z;
};

struct int2 {
    int x, y;
};

struct Sensor {
private:
    double3 pos;
    int index;
    std::complex<double> *output;
public:
    Sensor(double x, double y, double z, int index) : index(index) {
        pos.x = x;
        pos.y = y;
        pos.z = z;
    };

    Sensor() : pos(), index() {};

    friend std::ostream &operator<<(std::ostream &os, const Sensor obj) {
        os << "Element " << obj.index << " is at (x, y, z): (" << obj.get_pos().x << ", " << obj.get_pos().y << ", "
           << obj.get_pos().z << ")";
        return os;
    }

    double3 get_pos() const {
        return this->pos;
    }

    void set_pos_and_index(double pos_x, double pos_y, double pos_z, int idx) {
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
    std::complex<double> *steering_vector;
    int2 antenna_dim;

    ~Array() {
        free(antenna_elements);
        free(steering_vector);
    }

    /** Initialise the array with a specified reference element and point in space, with each sensor equidistantly positioned (by distance length units) in a line parallel to the x-axis
     *
     * @param ref_element the index of the array element considered to be the reference
     * @param distance the distance by which the array elements are separated from each other
     * @param ref_point the point where ref_element is located
     */
    Array(int2 ref_element, double3 ref_point, double d_x = 0.07f, double d_y = 0.07f, int rows = 4, int cols = 6)
            : ref_element(ref_element), ref_point(ref_point) {
        antenna_elements = (Sensor *) malloc(rows * cols * sizeof(Sensor));
        steering_vector = (std::complex<double> *) malloc(rows * cols * sizeof(std::complex<double>));
        antenna_dim = {rows, cols};
        double distance_x, distance_y, electrical_angle, distance; // distance to ref_element
        int offset;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                offset = i + rows * j;
                // this case is needed to match the array setup, as one column of the array was not functional at the time of measurement
                if (j >= 3) {
                    distance_x = (j - ref_element.x + 1) * d_x;
                } else {
                    distance_x = (j - ref_element.x) * d_x;
                }
                distance_y = (i - ref_element.y) * d_y;
                distance = sqrt(pow(distance_x, 2) + pow(distance_y, 2));
                electrical_angle = -INCIDENT_FREQUENCY / SPEED_OF_LIGHT * distance * cos(INCIDENT_ANGLE);
                steering_vector[offset] = std::complex<double>(cos(electrical_angle), sin(electrical_angle));
                antenna_elements[offset].set_pos_and_index(ref_point.x + distance_x, ref_point.y + distance_y,
                                                           ref_point.z,
                                                           offset); // place elements along line parallel to x-axis for now
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

    Sensor get_element_from_matrix_indices(int row, int col) const {
        return antenna_elements[row + col * antenna_dim.x];
    }

    friend std::ostream &operator<<(std::ostream &os, const Array &obj) {
        for (int i = 0; i < M; i++) {
            os << obj.antenna_elements[i] << std::endl << "Steering coefficient: " << obj.steering_vector[i].real()
               << " + " << obj.steering_vector[i].imag() << "i" << std::endl << std::endl;
        }
        return os;
    }
};


void
execute_cleanup(std::complex<double> *weights, std::complex<double> *array_output, std::complex<double> *mul_result,
                Array *array);

void assign_array_output(const int length, std::complex<double> *output, std::vector<double> data[20],
                         std::complex<double> *steering_vector, const int &time_step);

/** Host wrapper function for summing the cuDoubleComplexes in an array. Runs arr_sum_GPU to get block sums and then
 * calculates the sum of the block sums and returns that value as a cuDoubleComplex.
 *
 * @param length: integer length of the array to be summed
 * @param a: array of cuDoubleComplex to be summed
 * @param num_blocks: integer holding the number of block sums
 * @return cuDoubleComplex holding the sum of the array a
 */
std::complex<double> arr_sum(const int length, const std::complex<double> *a) {
    std::complex<double> result(0, 0);
    for (int i = 0; i < length; i++) result += a[i];
    return result;
}

/** Calculates the response of the beamformer at a certain time instance, y(t) = w^Hx(t)
 *
 * @param length the number of array elements there are.
 * @param weights the weights to be applied to the output of each array element
 * @param array_output the output from each array element
 * @return the accumulated response of the beamformer to the signal
 */
std::complex<double>
response(int length, std::complex<double> *weights, const std::complex<double> *array_output) {
    std::complex<double> res;
    for (int i = 0; i < length; i++) res += std::conj(weights[i]) * array_output[i];
    return res;
}

/** Calculate the absolute value of a complex number (the complex norm). Denoted |z| in mathematics, where z is a complex number
 *
 * @param z : A cuDoubleComplex whose absolute value one wants to calculate
 * @return The absolute value (complex norm) of the complex number z
 */
double squared_absolute(std::complex<double> z) {
    return std::pow(z.real(), 2) + std::pow(z.imag(), 2);
}

/** Calculates the euclidean length of an array of complex numbers
 *
 * @param length: size (number of elements) of the array
 * @param arr: the array to be normed
 * @return The euclidean norm of the array
 */
double calc_arr_length(const int length, const std::complex<double> *arr) {
    double res = 0;
    for (int i = 0; i < length; i++)
        res += std::norm(arr[i]);
    return res;
}


/** Calculates the optimal set of weights for the array elements from the direction vector
 *
 * @param a_theta The direction vector of the desired signal
 * @param length The length of the array (number of array elements)
 * @param weights The array where the calculated weights are to be stored
 */
void calc_weights_from_direction(const std::complex<double> *a_theta, const int length, std::complex<double> *weights) {
    double vector_norm = calc_arr_length(length, a_theta);
    for (int i = 0; i < length; i++) {
        weights[i] = a_theta[i] / vector_norm;
    }
}


/** Function to clean up the shared memory and close input file
 *
 * @param weights
 * @param array_output
 * @param mul_result
 * @param array
 * @param file
 */
void
execute_cleanup(std::complex<double> *weights, std::complex<double> *array_output, std::complex<double> *mul_result,
                Array *array,
                std::ifstream &file) {
    free(weights);
    free(array_output);
    free(mul_result);
    free(array);
    file.close();
}

int main() {
    // declare variables
    int num_time_steps = 4'194'304;
    Array *array;
    std::complex<double> *mul_result, *weights, *array_output;
    double power_output;
    std::cout << "testing" << std::endl;

    // Reading data from disk into contiguous block
    std::ifstream in("data_files/2022-01-11.txt");
    std::vector<double> data[M];
    for (auto &m: data)
        m.resize(num_time_steps);
    for (int j = 0; j < num_time_steps; j++) {
        for (auto &i: data) {
            in >> i[j];
        }
    }

    // allocate memory for variables
    weights = (std::complex<double> *) malloc(M * sizeof(std::complex<double>));
    array_output = (std::complex<double> *) malloc(M * sizeof(std::complex<double>));
    mul_result = (std::complex<double> *) malloc(M * sizeof(std::complex<double>));

    // Step 1: calculate steering vector for array
    array = new Array({0, 0}, {0, 0, 0});

    // Step 2: calculate weights using the steering vector
    calc_weights_from_direction(array->steering_vector, M, weights);

//    cudaMemcpy(&weights_h, &weights, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    auto start = clock();
    for (int t = 0; t < num_time_steps; t++) {
        // Step 3: Get array output by multiplying the steering vector with measured voltage
        assign_array_output(M, array_output, data, array->steering_vector, t);


        // Step 4: Calculate response by multiplying the array output with the calculated weights
        std::complex<double> res = response(M, weights, array_output);

        // Step 5: Calculate the Power output
        power_output = squared_absolute(res);

    }
    auto end = clock();
    clock_t duration = end - start;
    std::cout << "Processing took " << (double) duration / CLOCKS_PER_SEC << " seconds" << std::endl
              << "Processing rate is " << num_time_steps * 20 / ((double) duration / CLOCKS_PER_SEC)
              << " samples per second";
    execute_cleanup(weights, array_output, mul_result, array, in);
/**
 * Test code Below
 */
    // Test Array constructor
//    double3 ref_p({0, 0, 0});
//    Array *array;
//    cudaMallocManaged(&array, sizeof(Array));
//    array = new Array({2, 1}, ref_p);
//    std::cout << *array;
//    execute_cleanup(weights, array_output, mul_result, array, in);

    // initialize arrays
//    for (int i = 0; i < M; i++) {
//        mul_result[i] = make_cuDoubleComplex(2.0f, 3.0f);
//        weights[i] = make_cuDoubleComplex(2.0f, 5.0f);
//        array_output[i] = make_cuDoubleComplex(3.0f, 4.0f);
//    }

    // run test of array multiplication
//    test_arr_mul(M, mul_result, weights, array_output);
//
//    // run test of array sum
//    cuDoubleComplex sum = arr_sum(M, mul_result);
//    std::cout << cuCreal(sum) << " + " << cuCimag(sum) << "i";

    // run test of response function
//    cuDoubleComplex test = response(M, weights, array_output);
//    std::cout << "result of response function: " << cuCreal(test) << " + " << cuCimag(test) << "i" << std::endl;

    // test squared_absolute function
//    cuDoubleComplex z = make_cuDoubleComplex(26, -7);
//    double absolute = squared_absolute(z);
//    std::cout << "absolute of " << cuCreal(z) << cuCimag(z) << "i is " << absolute << std::endl;

    // test the function calculating the length of an array
//    calc_arr_length(M, weights);

    return 0;
}

void assign_array_output(const int length, std::complex<double> *output, std::vector<double> data[20],
                         std::complex<double> *steering_vector, const int &time_step) {
    for (int m = 0; m < length; m++) {
        output[m] = data[m][time_step] * steering_vector[m];
    }
}
