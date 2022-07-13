# GPU Beamforming

A place for code related to thesis work on beamforming performed on the GPU. Feel free to fork if you want to modify the implementations.

## Practical information
The code is developed in CUDA 11.0 on the softEW lab computer equipped with an _NVIDIA GeForce RTX 2080 Super_ GPU card.
The system is located in room A0441, but can be reached remotely from the ew-dev environment with this command:
`ssh <YOUR CORP USERNAME>@10.23.138.16 -Y`.
The -Y flag enables graphical interfaces such as file explorers, code editors, internet browsers etc.
If no graphical interfaces are needed, the -Y flag can be omitted.ĸ

## Implementations
The repo includes two beamformers; one Bartlett and one MVDR. Both are implemented in main.cu and are run from a
common interface in the main function. To run one or the other, switch the commented line in the following section
of the main function.

E.g. switching from executing Bartlett to MVDR:

```c++
    cudaMemcpy(data_d, data, M * num_time_steps * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    execute_bartlett_beamforming(array, weights, num_time_steps, power_output, data_d);
//    execute_mvdr_beamforming(array, weights, num_time_steps, power_output, data_d);

    execute_cleanup(weights, array, in);
```
```c++
    cudaMemcpy(data_d, data, M * num_time_steps * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

//    execute_bartlett_beamforming(array, weights, num_time_steps, power_output, data_d);
    execute_mvdr_beamforming(array, weights, num_time_steps, power_output, data_d);

    execute_cleanup(weights, array, in);
```

Another important note is the array geometry. This implementation uses the geometry specified by the WiDAR receiver
in the PADIC project, since their data is used for evaluation in the thesis. The geometry is a uniform planar array
of 4x6 antennas spaced 0.07m vertically and horizontally. However, since one column of antennas did not take any
samples at the time of measurement, it is skipped in the array constructor. The data files used are structured
with a sample per row with each column representing an array element. With zero-indexing on a row of inputs, the
elements in the array are placed as follows:

<table>
    <tr>
        <td> 0 </td><td> 4 </td><td> 8 </td> <td> _ </td><td> 12 </td><td> 16 </td>
    </tr>
    <tr>
        <td> 1 </td><td> 5 </td><td> 9 </td> <td> _ </td><td> 13 </td><td> 17 </td>
    </tr>
    <tr>
        <td> 2 </td><td> 6 </td><td> 10 </td><td> _ </td><td> 14 </td><td> 18 </td>
    </tr>
    <tr>
        <td> 3 </td><td> 7 </td><td> 11 </td><td> _ </td><td> 15 </td><td> 19 </td>
    </tr>
</table>

The steering vector (matrix) calculated for the planar array is based on the following website:
https://www.antenna-theory.com/arrays/weights/twoDuniform.php (open in CORP environment).

The file device.cu holds utility functions for taking the reciprocal of each element of a vector and for
extracting the power output of the beamformer in a certain time instance.   

Owner: Oscar Lindgren (u080334)