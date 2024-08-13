# Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression

This is the implementation of the paper:
> Peiyu Liu, Ze-Feng Gao, Wayne Xin Zhao, Yipeng Ma, Tao Wang and Ji-Rong Wen. Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression
*Updates*:

* [February 29] We update the README and initial code.

---
## Code for paper
### Prepare the packages
``` shell
wget https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip ./${eigen_path}
```

### Install
1. All of the dependent packages can be found in the ``requirements.txt``

2. Edit the environment variable
```shell
export DECOQUANT_PATH=[your path]
```

2. download ``eigen`` package
```shell
export EIGEN_PATH=[your path]
wget https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip $EIGEN_PATH/
cd $EIGEN_PATH
unzip eigen-master.zip
export EIGEN_PATH=$EIGEN_PATH/eigen-master
```

3. Edit the setup.py file
```python
from __future__ import print_function
from setuptools import setup, find_packages
from distutils.core import Extension
import os

eigen_path = os.environ.get('EIGEN_PATH')
svd_module = Extension(name='svd_module', 
                           sources=['bdcsvd.cpp'],
                           include_dirs=[{eigen_path},
                                         r'/usr/local/lib/python3.10/dist-packages/pybind11/include'], # this should be your own path
                           )

setup(ext_modules=[svd_module])
```

4. Build the module:
```shell
cd $DECOQUANT_PATH/external_modules/bdcsvd
python setup.py install
# test the import module
cd $DECOQUANT_PATH/icl
python test_module.py
```

### Table-1
```shell
cd language_model
# bash run.sh

# w/o quantize KV cache
run fp16_opt1_3 [gpu] fp16 /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"

# w KV cache, quantzation setting: W16A_
# naive_W16A_
run naive_8bit_opt1_3_W16A8 [gpu] base_naive_kv /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"
run naive_4bit_opt1_3_W16A4 [gpu] base_naive_kv /media/liupeiyu/opt-1.3b 4 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"
run naive_2bit_opt1_3_W16A2 [gpu] base_naive_kv /media/liupeiyu/opt-1.3b 2 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"
# # ours_W16A_
run ours_8bit_opt1_3_W16A8 [gpu] base_ours_kv /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"
run ours_4bit_opt1_3_W16A4 [gpu] base_ours_kv /media/liupeiyu/opt-1.3b 4 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"
run ours_2bit_opt1_3_W16A2 [gpu] base_ours_kv /media/liupeiyu/opt-1.3b 2 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"

# w KV cache, quantzation setting: W_A_
# # naive_W_A_
run naive_8bit_opt1_3_W8A8 [gpu] base_naive_kv /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
run naive_4bit_opt1_3_W4A4 [gpu] base_naive_kv /media/liupeiyu/opt-1.3b 4 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
run naive_2bit_opt1_3_W2A2 [gpu] base_naive_kv /media/liupeiyu/opt-1.3b 2 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
# # ours_W_A_
run ours_8bit_opt1_3_W8A8 [gpu] base_ours_kv /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
run ours_4bit_opt1_3_W4A4 [gpu] base_ours_kv /media/liupeiyu/opt-1.3b 4 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
run ours_2bit_opt1_3_W2A2 [gpu] base_ours_kv /media/liupeiyu/opt-1.3b 2 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"

# smoothqaunt_W_A_
run sq_8bit_opt1_3_W8A8 [gpu] smoothquant /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
run sq_4bit_opt1_3_W4A4 [gpu] smoothquant /media/liupeiyu/opt-1.3b 4 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
run sq_2bit_opt1_3_W2A2 [gpu] smoothquant /media/liupeiyu/opt-1.3b 2 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"

# smoothqaunt_W_A_
run sq_8bit_opt67_W8A8 [gpu] smoothquant /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
run sq_4bit_opt67_W4A4 [gpu] smoothquant /media/liupeiyu/opt-1.3b 4 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
run sq_2bit_opt67_W2A2 [gpu] smoothquant /media/liupeiyu/opt-1.3b 2 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"

```

### Table-3
```shell
cd icl
# bash run.sh

# w/o KV cache, "--num_plaintext_demonstrations" indicates different length of prompts
run_ori eval_${data_name}_ori_0 [gpu] --num_plaintext_demonstrations=0\ --dataset=${data_name}\ --model_path=${model_name}
run_ori eval_${data_name}_ori_2 [gpu] --num_plaintext_demonstrations=2\ --dataset=${data_name}\ --model_path=${model_name}
run_ori eval_${data_name}_ori_10 [gpu] --num_plaintext_demonstrations=10\ --dataset=${data_name}\ --model_path=${model_name}

# w KV cache, RTN, "--num_bits" indicates the number of bits
run eval_${data_name}_naive_quant2_demo2 [gpu] --num_bits=2\ --exp=naive\ --num_plaintext_demonstrations=2\ --dataset=${data_name}\ --model_path=${model_name}
run eval_${data_name}_naive_quant4_demo2 [gpu] --num_bits=4\ --exp=naive\ --num_plaintext_demonstrations=2\ --dataset=${data_name}\ --model_path=${model_name}
run eval_${data_name}_naive_quant2_demo10 [gpu] --num_bits=2\ --exp=naive\ --num_plaintext_demonstrations=10\ --dataset=${data_name}\ --model_path=${model_name}
run eval_${data_name}_naive_quant4_demo10 [gpu] --num_bits=4\ --exp=naive\ --num_plaintext_demonstrations=10\ --dataset=${data_name}\ --model_path=${model_name}

# w KV cache，ours
run eval_${data_name}_ours_quant2_demo2 [gpu] --num_bits=2\ --exp=ours\ --num_plaintext_demonstrations=2\ --dataset=${data_name}\ --model_path=${model_name}
run eval_${data_name}_ours_quant4_demo2 [gpu] --num_bits=4\ --exp=ours\ --num_plaintext_demonstrations=2\ --dataset=${data_name}\ --model_path=${model_name}
run eval_${data_name}_ours_quant4_demo10 [gpu] --num_bits=4\ --exp=ours\ --num_plaintext_demonstrations=10\ --dataset=${data_name}\ --model_path=${model_name}
run eval_${data_name}_ours_quant2_demo10 [gpu] --num_bits=2\ --exp=ours\ --num_plaintext_demonstrations=10\ --dataset=${data_name}\ --model_path=${model_name}
```


