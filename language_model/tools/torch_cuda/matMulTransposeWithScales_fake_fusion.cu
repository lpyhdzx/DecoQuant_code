#include <cuda_fp16.h>
#include <cublas_v2.h>

extern "C" {
    __global__ void scaleMatrix(const __half* A, const __half* scales, __half* A_scaled, int m, int n) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < m && col < n) {
            A_scaled[row * n + col] = __hmul(A[row * n + col], scales[row]);
        }
    }
}

void launchmatMulTransposeWithScalesCublas(const __half* A, const __half* B, __half* C, const __half *scales, int m, int n, int k) {
    // Scale matrix A
    __half* A_scaled;
    cudaMalloc(&A_scaled, sizeof(__half) * m * n);
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    scaleMatrix<<<grid, block>>>(A, scales, A_scaled, m, n);

    // Setup cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication using cuBLAS
    const __half alpha = __float2half(1.0);
    const __half beta = __float2half(0.0);
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, A_scaled, m, B, n, &beta, C, m);

    // Cleanup
    cudaFree(A_scaled);
    cublasDestroy(handle);
}
