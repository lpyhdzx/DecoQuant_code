import time
import torch
# import my_cuda_extension
from torch.utils.cpp_extension import load


cuda_module = load(name="matMulTransposeWithScales", 
                   sources=["/home/liupeiyu/flan-eval/tools/torch_cuda/matMulTransposeWithScales.cpp","/home/liupeiyu/flan-eval/tools/torch_cuda/matMulTransposeWithScales.cu"])

# A = torch.rand(4, 3, device='cuda', dtype=torch.float32)
# B = torch.rand(3, 4, device='cuda', dtype=torch.float32)
A = torch.tensor([[ 1.,  2.,  3.],
              [ 4.,  5.,  6.],
              [ 7.,  8.,  9.],
              [10., 11., 12.]], device='cuda',dtype=torch.float16)

B = torch.tensor([[ 1.,  2.,  3.,  4.],
              [ 5.,  6.,  7.,  8.],
              [ 9., 10., 11., 12.]], device='cuda',dtype=torch.float16)

cuda_c = torch.rand((4,4), dtype=torch.float16,device='cuda') 

scales = torch.tensor([1., 2., 3.], device='cuda', dtype=torch.float16)
def run_cuda():
    cuda_module.torch_launchMatMulTransposeWithScales(A, B, cuda_c, scales)
start=time.time()
for _ in range(1000):
    run_cuda()
print(f"cuda time: {time.time()-start}")
print(cuda_c)
def run_torch():
    res = torch.matmul(A.mul(scales),B).T
    return res
start=time.time()
# for _ in range(1000):
res = run_torch()
print(f"torch time: {time.time()-start}")
print(res)