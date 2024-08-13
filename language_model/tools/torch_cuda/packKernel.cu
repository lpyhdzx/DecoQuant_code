#include <cuda.h>
__global__ void pack_kernel(int32_t* packed, const int32_t* unpacked, int n_bits, int values_per_int32, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int row = idx / values_per_int32;
    int col = idx % values_per_int32;

    atomicOr(&packed[row], (unpacked[idx] & ((1 << n_bits) - 1)) << (col * n_bits));
}

// __global__ void unpack_kernel(const int32_t* packed, int32_t* unpacked, int n_bits, int values_per_int32, int total_elements) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= total_elements) return;

//     int row = idx / values_per_int32;
//     int col = idx % values_per_int32;

//     unpacked[idx] = (packed[row] >> (col * n_bits)) & ((1 << n_bits) - 1);
// }
__global__ void unpack_kernel(const int32_t* packed, int32_t* unpacked, int n_bits, int values_per_int32, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int row = idx / values_per_int32;
    int col = idx % values_per_int32;

    // Perform an unsigned shift and then cast to signed int
    unsigned int shifted = static_cast<unsigned int>(packed[row]) >> (col * n_bits);
    int32_t masked = shifted & ((1 << n_bits) - 1);

    // Adjust for sign if n_bits represents the full width of the type
    if (n_bits == sizeof(int32_t) * 8) {
        unpacked[idx] = masked;
    } else {
        // Check if the sign bit is set, and extend the sign if necessary
        int32_t sign_bit = 1 << (n_bits - 1);
        unpacked[idx] = (masked ^ sign_bit) - sign_bit;
    }
}

void launchpackKernel(int32_t* packed, const int32_t* unpacked, int n_bits, int values_per_int32, int total_elements) {
    
    // dim3 grid((k + 15) / 16, (m + 15) / 16); // Adjusted for a block size of 16x16
    // dim3 block(16, 16);
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    pack_kernel<<<blocks_per_grid, threads_per_block>>>(packed, unpacked, n_bits, values_per_int32, total_elements);
    cudaDeviceSynchronize(); // Ensure kernel execution completes
}
void launchunpackKernel(const int32_t* packed, int32_t* unpacked, int n_bits, int values_per_int32, int total_elements) {

    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    unpack_kernel<<<blocks_per_grid, threads_per_block>>>(packed, unpacked, n_bits, values_per_int32, total_elements);
    cudaDeviceSynchronize(); // Ensure kernel execution completes
}