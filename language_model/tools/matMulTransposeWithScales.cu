#include <stdio.h>
extern "C" {
    // __global__ void matMulTransposeWithScales(float *A, float *B, float *C, float *scales, int m, int n, int k) {
    //     int col = blockIdx.x * blockDim.x + threadIdx.x;
    //     int row = blockIdx.y * blockDim.y + threadIdx.y;

    //     if (row < m && col < k) {
    //         float sum = 0;
    //         for (int i = 0; i < n; ++i) {
    //             sum += A[row * n + i] * scales[i] * B[i * k + col];
    //         }
    //         // C[col * m + row] = sum; // Transposing while writing the result
    //         C[row * k + col] = sum; // Without transposing
    //     }
    // }
    __global__ void matMulTransposeWithScales(float *A, float *B, float *C, float *scales, int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < k) {
        float sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * scales[i] * B[i * k + col];
        }
        C[row * k + col] = sum; 
    }
}
}

extern "C" {
    void launchMatMulTransposeWithScales(float *A, float *B, float *C, float *scales, int m, int n, int k) {
        // dim3 grid((k + 15) / 16, (m + 15) / 16); // Adjusted for a block size of 16x16
        // dim3 block(16, 16);
        dim3 dimBlock(16, 16);
        dim3 dimGrid(1, 1); // Since the output matrix C is 4x4

        float a_host, b_host, c_host;
        cudaMemcpy(&a_host, A, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b_host, B, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&c_host, C, sizeof(float), cudaMemcpyDeviceToHost);

        printf("Debug: A[0] = %f, B[0] = %f, C[0] = %f\n", a_host, b_host, c_host);


        // for (int i =0; i<n; ++i) {
        //     printf("%f",A[i]);
        // }    
        matMulTransposeWithScales<<<dimGrid, dimBlock>>>(A, B, C, scales, m, n, k);
        cudaDeviceSynchronize(); // Ensure kernel execution completes
    }
}
