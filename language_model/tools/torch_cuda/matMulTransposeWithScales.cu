#include <cuda_fp16.h>

#define TILE_DIM 32
#define BLOCK_ROWS 32

__global__ void matMulTransposeWithScalesCublas(const __half* A, const __half* B, __half* C, const __half *scales, int m, int n, int k) {
    __shared__ __half tile_A[TILE_DIM][TILE_DIM];
    __shared__ __half tile_B[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_ROWS + ty;
    int col = bx * TILE_DIM + tx;

    __half acc = __float2half(0.0);

    for (int t = 0; t < (n + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load A and B into shared memory, applying scaling to A
        if (t * TILE_DIM + tx < n && row < m) {
            tile_A[ty][tx] = __hmul(A[row * n + t * TILE_DIM + tx], scales[row]);
        } else {
            tile_A[ty][tx] = __float2half(0.0);
        }

        if (t * TILE_DIM + ty < n && col < k) {
            tile_B[ty][tx] = B[(t * TILE_DIM + ty) * k + col];
        } else {
            tile_B[ty][tx] = __float2half(0.0);
        }

        __syncthreads();

        // Compute product for this tile
        for (int i = 0; i < TILE_DIM; ++i) {
            acc = __hadd(acc, __hmul(tile_A[ty][i], tile_B[i][tx]));
        }

        __syncthreads();
    }

    // Write the result
    if (row < m && col < k) {
        C[col * m + row] = acc; // Transposing while writing the result
    }
}
void launchmatMulTransposeWithScalesCublas(const __half* A, const __half* B, __half* C, const __half* scales, int m, int n, int k) {

    dim3 dimBlock(8,8);
    dim3 dimGrid((1000 + dimBlock.x - 1) / dimBlock.x, (4096 + dimBlock.y - 1) / dimBlock.y);
    // dim3 blockSize(TILE_DIM, BLOCK_ROWS);
    // dim3 numBlocks((k + TILE_DIM - 1) / TILE_DIM, (m + BLOCK_ROWS - 1) / BLOCK_ROWS);
    // matMulTransposeWithScalesCublas<<<numBlocks, blockSize>>>(A, B, C, scales, m, n, k);
    matMulTransposeWithScalesCublas<<<dimGrid, dimBlock>>>(A, B, C, scales, m, n, k);
    cudaDeviceSynchronize(); // Ensure kernel execution completes
}
