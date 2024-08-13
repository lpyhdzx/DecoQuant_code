// extern "C" int testFunction(int a) {
//     // Perform some operations
//     return a;  // Return the input value or some status code
// // }
// #include <stdio.h>
// extern "C" {
//     __global__ void testKernel() {
//         printf("Hello from the GPU!\n");
//         int col = blockIdx.x * blockDim.x + threadIdx.x;
//         int row = blockIdx.y * blockDim.y + threadIdx.y;
//         printf("Hello from the GPU!\n: %f\n", row);
//         printf("Hello from the GPU!\n: %f\n", col);

//     }
// }

// int main() {
//     testKernel<<<1, 1>>>();
//     cudaDeviceSynchronize();
//     return 0;
// }

#include <stdio.h>

__global__ void testKernel() {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    printf("Hello from the GPU! Row: %d, Col: %d\n", row, col);
}

extern "C" {
    void launchTestKernel(int gridDimX, int gridDimY, int blockDimX, int blockDimY) {
        dim3 grid(gridDimX, gridDimY);
        dim3 block(blockDimX, blockDimY);
        testKernel<<<grid, block>>>();
        cudaDeviceSynchronize();  // Synchronize to ensure kernel execution is finished
    }
}

