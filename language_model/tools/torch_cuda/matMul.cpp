#include <torch/extension.h>

#include <cuda_fp16.h>

void launchmatMul(const __half* A, const __half* B, __half* C, int m, int n, int k);


// torch::Tensor torch_launchMatMulTransposeWithScales(torch::Tensor A, torch::Tensor B, torch::Tensor scales) {
void torch_launchmatMul(
                                    const torch::Tensor &A, 
                                    const torch::Tensor &B, 
                                    torch::Tensor &C) {
    // auto C = torch::zeros_like(A.matmul(B)); // or another appropriate way to create C

    launchmatMul((const __half *)A.data_ptr(), 
                                    (const __half *)B.data_ptr(), 
                                    (__half *)C.data_ptr(), 
                                    A.size(0), A.size(1), B.size(1));

    // return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launchmatMul",
    &torch_launchmatMul, 
    "Matrix multiplication with scales kernel launch");
}
