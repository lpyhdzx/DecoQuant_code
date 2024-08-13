from __future__ import print_function
from setuptools import setup
from distutils.core import Extension
import os

eigen_path = os.environ.get('EIGEN_PATH')
svd_module = Extension(name='svd_module',
                           sources=['bdcsvd.cpp'],
                           include_dirs=[eigen_path,
                                         r'/usr/local/lib/python3.10/dist-packages/pybind11/include'], # this should be your own path
                           )

setup(ext_modules=[svd_module])