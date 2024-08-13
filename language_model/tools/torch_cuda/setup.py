# # setup.py
# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup(
#     name='my_cuda_extension',
#     ext_modules=[
#         CUDAExtension('my_cuda_extension', [
#             'my_extension.cpp',
#             'matMulTransposeWithScales.cu',
#         ])
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_cuda_extension',
    ext_modules=[
        CUDAExtension('my_cuda_extension', [
            '/home/liupeiyu/flan-eval/tools/torch_cuda/my_extension.cpp',    # C++ source file
            '/home/liupeiyu/flan-eval/tools/torch_cuda/matMulTransposeWithScales.cu',  # CUDA source file
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
