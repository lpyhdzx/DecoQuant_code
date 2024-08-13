#include <stdio.h>
extern "C" {
    __global__ void matMulTransposeWithScales(const float* A, const float* B, float* C, const float *scales, int m, int n, int k) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < m && col < k) {
            float sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += A[row * n + i] * scales[i] * B[i * k + col];
            }
            C[col * m + row] = sum; // Transposing while writing the result
            // C[row * k + col] = sum; // Without transposing
        }
    }
}

void launchMatMulTransposeWithScales(const float* A, const float* B, float* C, const float* scales, int m, int n, int k) {
    dim3 grid((k + 15) / 16, (m + 15) / 16); // Adjusted for a block size of 16x16
    dim3 block(16, 16);

    matMulTransposeWithScales<<<grid, block>>>(A, B, C, scales, m, n, k);
    cudaDeviceSynchronize(); // Ensure kernel execution completes
}


// extern "C" {
//     void launchMatMulTransposeWithScales(float *A, float *B, float *C, float *scales, int m, int n, int k) {
//         dim3 grid((k + 15) / 16, (m + 15) / 16); // Adjusted for a block size of 16x16
//         dim3 block(16, 16);
  
//         matMulTransposeWithScales<<<grid, block>>>(A, B, C, scales, m, n, k);
//         cudaDeviceSynchronize(); // Ensure kernel execution completes
//     }
// }
