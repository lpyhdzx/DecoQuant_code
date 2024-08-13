#include <stdio.h>
#include <cuda_fp16.h>

extern "C" {
    __global__ void matMulTransposeWithScales(const __half* A, const __half* B, __half* C, const __half *scales, int m, int n, int k) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < m && col < k) {
            __half sum = __float2half(0.0);
            for (int i = 0; i < n; ++i) {
                sum = __hadd(sum, __hmul(__hmul(A[row * n + i], scales[row]), B[i * k + col]));
            }
            C[col * m + row] = sum; // Transposing while writing the result
        }
    }
}

void launchMatMulTransposeWithScales(const __half* A, const __half* B, __half* C, const __half* scales, int m, int n, int k) {
    dim3 grid((k + 15) / 16, (m + 15) / 16); // Adjusted for a block size of 16x16
    dim3 block(16, 16);

    matMulTransposeWithScales<<<grid, block>>>(A, B, C, scales, m, n, k);
    // dim3 dimBlock(16,16);
    // dim3 dimGrid((1000 + dimBlock.x - 1) / dimBlock.x, (4096 + dimBlock.y - 1) / dimBlock.y);
    // matMulTransposeWithScales<<<dimGrid, dimBlock>>>(A, B, C, scales, m, n, k);
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
