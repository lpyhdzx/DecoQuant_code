#include <stdio.h>
#include <cuda_fp16.h>

extern "C" {
    __global__ void matMul(const __half* A, const __half* B, __half* C, int m, int n, int k) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < m && col < k) {
            __half sum = __float2half(0.0);
            for (int i = 0; i < n; ++i) {
                sum = __hadd(sum, __hmul(A[row * n + i], B[i * k + col]));
            }
            C[row * k + col] = sum; // Without transposing
        }
    }
}

void launchmatMul(const __half* A, const __half* B, __half* C, int m, int n, int k) {
    dim3 grid((k + 15) / 16, (m + 15) / 16); // Adjusted for a block size of 16x16
    dim3 block(16, 16);

    matMul<<<grid, block>>>(A, B, C, m, n, k);
    cudaDeviceSynchronize(); // Ensure kernel execution completes
}
