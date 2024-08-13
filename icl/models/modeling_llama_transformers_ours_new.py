import sys
from transformers.utils import logging
import torch
import torch.utils.checkpoint
from torch import nn
decquant_path = os.environ.get('DECOQUANT_PATH')

sys.path.append(eigen_path+"/icl")
from tools.Matrix2MPO_beta import MPO
from .fake_quant import quantize_activation_per_token_absmax_real,quantize_activation_per_token_absmax,quantize_activation_per_token_absmax_test
from torch.utils.cpp_extension import load
import os


matmul_module = load(name="matMul",
                     sources=[decquant_path + "/icl/tools/torch_cuda/matMul.cpp",decquant_path + "/icl/tools/torch_cuda/matMul.cu"])
cublass_module = load(name="matMulTransposeWithScalesCublas",
                      sources=[decquant_path + "/icl/tools/torch_cuda/matMulTransposeWithScales.cpp",decquant_path + "/icl/tools/torch_cuda/matMulTransposeWithScales.cu"])

logger = logging.get_logger(__name__)


def dequantize(exp, tensor_set0, tensor_set1, scales, newshape, mpo=None):
    if exp == "torch":
        tensor_set0.mul_(scales)
        rec2 = torch.zeros((tensor_set0.shape[0], tensor_set1.shape[1]),dtype=torch.float16).cuda()
        matmul_module.torch_launchmatMul(tensor_set0, tensor_set1, rec2)
        return rec2.reshape(newshape)
    elif exp == "cuda":
        rec2 = torch.empty((tensor_set0.shape[0], tensor_set1.shape[1]),dtype=torch.float16).cuda()
        cublass_module.torch_launchmatMulTransposeWithScalesCublas(tensor_set0, tensor_set1, rec2, scales)
        return rec2.view(newshape)


SHAPE_CONFIG = {
    "attention":([4,4,8,8,4],[4,8,8,4,4]),
    "gate_proj":([4,4,43,4,4],[4,4,8,8,4]),
    "down_proj":([4,4,8,8,4],[4,4,43,4,4]),
    "up_proj":([4,4,43,4,4],[4,4,8,8,4])
}
# KV_SHAPE = ([1,1],[2048,1]) # llama-7b
# KV_SHAPE = ([1,1],[256,16]) # llama-7b
KV_SHAPE = ([1,1],[128,16]) # OPT-1.3
# KV_SHAPE = ([1,1],[256,16]) # OPT-6.7
# KV_SHAPE = ([1,1],[64,32]) # OPT-1.3

THRES = 0.4
SMART_CACHE = True
class KVStates(nn.Module):
    def __init__(self,typestates,D=4096,r=4,local_thres=0,num_bits=16, col_dim=500, exp=None):
        self.local_sta = None
        self.D = D
        self.r = r
        self.share_type_A_inv = None
        self.share_type_A = None
        self.local_thres = local_thres
        self.prefill_shape = typestates.shape # [bsz, nh, t, hd]
        self.bsz, self.nh, _, self.hd = self.prefill_shape
        self.num_bits = num_bits
        self.col_dim = col_dim
        self.after_scales = None
        self.before_scales = None
        self.exp = exp
        self.has_output = True
        self._init(typestates)
    def _init(self,hidden):
        self.device = hidden.device
        bsz,_,t,_ = hidden.shape # [bsz, nh, t, hd] [1,32,1,128]
        # assert self.bsz == 1
        if self.local_thres > 0:
            self.local_len = t % self.local_thres
        else:
            self.local_len = 0
        self.compress_len = t - self.local_len
        self.mpo_tensor_set = []
        self.cache_value = []

        self.values_per_int32 = 32 // self.num_bits
        
        self.mpo = MPO([self.compress_len // KV_SHAPE[0][0],1], KV_SHAPE[1], 100000)


        if self.compress_len > 0:
            self.local_sta = hidden[:,:,self.compress_len:,:]
            value = hidden[:,:,:self.compress_len,:].transpose(1,2).data.cpu().detach().numpy().reshape(self.compress_len,-1)
            mpo_tensor_set,_,_ = self.mpo.matrix2mpo(value)

            self.mpo_tensor_set = [torch.from_numpy(i) for i in mpo_tensor_set]

            self.mpo_tensor_set[0].squeeze_(0).squeeze_(0)
            self.scales = quantize_activation_per_token_absmax_test(self.mpo_tensor_set[0], self.num_bits)
            if self.has_output:
                print(f"Compression Ratio: {(self.scales.numel()*16+self.mpo_tensor_set[1].numel()*16+self.mpo_tensor_set[0].numel()*4)/(torch.from_numpy(value).numel()*16)}")
                self.has_output = False
            
            self.mpo_tensor_set[0].unsqueeze_(0)

            ### unpack
            self.mpo_tensor_set[0] = self.mpo_tensor_set[0].to(self.device).half()
            self.mpo_tensor_set[1] = self.mpo_tensor_set[1].to(self.device).half()

        else:
            self.local_sta = hidden

    def update(self,hidden):
        '''
        hidden: shape of [4,hidden_size], where 4 denotes the length.
        '''

        self.local_sta = hidden[:,:,self.compress_len:,:]
    def get_length(self):
        return self.local_sta.shape[2] + self.compress_len
    def get_hidden(self):
        reconstruct = self.mpo.mpo2matrix(self.mpo_tensor_set).reshape([1, -1, 32, self.D//32]).transpose(1,2).half() # 32 denotes head_num

        return torch.cat([reconstruct, self.local_sta], dim=2)