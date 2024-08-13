import torch
from torch.utils.cpp_extension import load
import math
pack_unpack_extension = load(
    name='pack_unpack_extension',
    sources=['/home/liupeiyu/flan-eval/tools/torch_cuda/packKernel.cpp', '/home/liupeiyu/flan-eval/tools/torch_cuda/packKernel.cu'],
    verbose=True
)

# Example usage
n_bits = 4
values_per_int32 = 32 // n_bits
original_matrix = torch.randint(-4, 4, (4096, 1000), dtype=torch.int32, device='cuda')

total_elements = original_matrix.numel()
packed_elements = math.ceil(total_elements / values_per_int32)

packed_matrix = torch.zeros(packed_elements, dtype=torch.int32, device='cuda')
unpacked_matrix = torch.zeros_like(original_matrix,dtype=torch.int32)

# Define grid and block sizes for CUDA
threads_per_block = 256
blocks_per_grid = (total_elements + (threads_per_block - 1)) // threads_per_block


# Example usage
pack_unpack_extension.torch_launchpackKernel(packed_matrix, original_matrix, n_bits,values_per_int32,total_elements)
print(f"Total elements: {packed_matrix.numel()}")
print("Values: \n")
print(packed_matrix)
pack_unpack_extension.torch_launchunpackKernel(packed_matrix, unpacked_matrix, n_bits,values_per_int32,total_elements)

print(f"Error: {torch.norm(original_matrix.float()-unpacked_matrix.float())}")