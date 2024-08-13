import time 
import numpy as np 
import torch 
from torch.utils.cpp_extension import load 

cuda_module = load(name="add2", sources=["add2.cpp", "add2.cu"], verbose=True) # c = a + b (shape: [n]) 

n = 1024 * 1024 
a = torch.rand(n, device="cuda:0") 
b = torch.rand(n, device="cuda:0") 
cuda_c = torch.rand(n, device="cuda:0") 
ntest = 10 
def show_time(func): 
    times = list() 
    res = list() # GPU warm up 
    for _ in range(10): 
        func() 
    for _ in range(ntest): # sync the threads to get accurate cuda running time 
        torch.cuda.synchronize(device="cuda:0") 
        start_time = time.time() 
        r = func() 
        torch.cuda.synchronize(device="cuda:0") 
        end_time = time.time() 
        times.append((end_time-start_time)*1e6) 
        res.append(r) 
    return times, res 
def run_cuda(): 
    cuda_module.torch_launch_add2(cuda_c, a, b, n) 
    return cuda_c 
def run_torch(): # return None to avoid intermediate GPU memory application # for accurate time statistics 
    a + b 
    return None 
print("Running cuda...") 
cuda_time, _ = show_time(run_cuda) 
print("Cuda time: {:.3f}us".format(np.mean(cuda_time))) 

print("Running torch...") 
torch_time, _ = show_time(run_torch) 
print("Torch time: {:.3f}us".format(np.mean(torch_time)))

