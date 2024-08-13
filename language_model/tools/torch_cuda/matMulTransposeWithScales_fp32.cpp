#include <torch/extension.h>

void launchMatMulTransposeWithScales(const float* A, const float* B, float* C, const float* scales, int m, int n, int k);


// torch::Tensor torch_launchMatMulTransposeWithScales(torch::Tensor A, torch::Tensor B, torch::Tensor scales) {
void torch_launchMatMulTransposeWithScales(
                                    const torch::Tensor &A, 
                                    const torch::Tensor &B, 
                                    torch::Tensor &C, 
                                    const torch::Tensor &scales) {
    // auto C = torch::zeros_like(A.matmul(B)); // or another appropriate way to create C

    launchMatMulTransposeWithScales((const float *)A.data_ptr(), 
                                    (const float *)B.data_ptr(), 
                                    (float *)C.data_ptr(), 
                                    (const float *)scales.data_ptr(), 
                                    // A.size(0), A.size(1), B.size(1));
                                    4,3,4);

    // return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launchMatMulTransposeWithScales",
    &torch_launchMatMulTransposeWithScales, 
    "Matrix multiplication with scales kernel launch");
}
