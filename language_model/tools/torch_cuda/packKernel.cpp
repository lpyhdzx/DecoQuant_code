#include <torch/extension.h>
#include <cuda_runtime.h>

// extern void pack_kernel(int32_t* packed, const int8_t* unpacked, int n_bits, int values_per_int32, int total_elements);
// extern void unpack_kernel(const int32_t* packed, int8_t* unpacked, int n_bits, int values_per_int32, int total_elements);

// void torch_pack(const torch::Tensor &unpacked, torch::Tensor &packed, int n_bits) {
//     int values_per_int32 = 32 / n_bits;
//     int total_elements = unpacked.numel();
//     int threads_per_block = 256;
//     int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

//     pack_kernel<<<blocks_per_grid, threads_per_block>>>(
//         packed.data_ptr<int32_t>(),
//         unpacked.data_ptr<int8_t>(),
//         n_bits,
//         values_per_int32,
//         total_elements
//     );
// }

// void torch_unpack(const torch::Tensor &packed, torch::Tensor &unpacked, int n_bits) {
//     int values_per_int32 = 32 / n_bits;
//     int total_elements = unpacked.numel();
//     int threads_per_block = 256;
//     int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

//     unpack_kernel<<<blocks_per_grid, threads_per_block>>>(
//         packed.data_ptr<int32_t>(),
//         unpacked.data_ptr<int8_t>(),
//         n_bits,
//         values_per_int32,
//         total_elements
//     );
// }

void launchpackKernel(int32_t* packed, const int32_t* unpacked, int n_bits, int values_per_int32, int total_elements);
void launchunpackKernel(const int32_t* packed, int32_t* unpacked, int n_bits, int values_per_int32, int total_elements);
// torch::Tensor torch_launchMatMulTransposeWithScales(torch::Tensor A, torch::Tensor B, torch::Tensor scales) {
void torch_launchpackKernel(
                            torch::Tensor &packed, 
                            const torch::Tensor &unpacked, 
                            int n_bits, 
                            int values_per_int32,
                            int total_elements){

    launchpackKernel((int32_t *)packed.data_ptr(), 
                        (const int32_t *)unpacked.data_ptr(), 
                        n_bits, values_per_int32, total_elements);

}
void torch_launchunpackKernel(
                            const torch::Tensor &packed, 
                            torch::Tensor &unpacked, 
                            int n_bits, 
                            int values_per_int32,
                            int total_elements){

    launchunpackKernel((const int32_t *)packed.data_ptr(), 
                        (int32_t *)unpacked.data_ptr(), 
                        n_bits, values_per_int32, total_elements);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launchpackKernel", &torch_launchpackKernel, "Packing N-bit values into int32");
    m.def("torch_launchunpackKernel", &torch_launchunpackKernel, "Packing N-bit values into int32");

}
