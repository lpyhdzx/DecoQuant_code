# -*- coding: utf-8 -*-
import numpy as np
import random
import torch.nn as nn
import torch
seed = 1234
random.seed(seed)
np.random.seed(seed)
import sys
import os
decquant_path = os.environ.get('DECOQUANT_PATH')

# sys.path.append(os.path.join(decquant_path,"/external_modules/mpo_c/build"))
# sys.path.append("/home/liupeiyu/flan-eval/mpo_c_2/build/lib.linux-x86_64-cpython-38")
sys.path.append(os.path.join(decquant_path,"external_modules/bdcsvd/build/lib.linux-x86_64-3.10"))


import svd_module

def power_iteration_svd(matrix, k, max_iter=100):
    """
    Computes SVD using power iteration.
    :param matrix: Input matrix
    :param k: Number of singular values and vectors
    :param max_iter: Maximum number of iterations
    :return: U, S, Vt matrices
    """
    U = np.empty((matrix.shape[0], k))
    S = np.empty(k)
    Vt = np.empty((k, matrix.shape[1]))
    
    for i in range(k):
        u = np.random.rand(matrix.shape[0])
        v = np.random.rand(matrix.shape[1])
        
        for _ in range(max_iter):
            v = np.dot(matrix.T, u)
            v /= np.linalg.norm(v)
            u = np.dot(matrix, v)
        
        sigma = np.dot(u, np.dot(matrix, v))
        U[:, i] = u
        S[i] = sigma
        Vt[i, :] = v
        
        matrix -= sigma * np.outer(u, v)
        
    return U, S, Vt

class MPO:
    def __init__(self, mpo_input_shape, mpo_output_shape, truncate_num, fix_rank=None):
        self.mpo_input_shape = mpo_input_shape
        self.mpo_output_shape = mpo_output_shape
        self.truncate_num = truncate_num
        self.num_dim = len(mpo_input_shape)
        self.mpo_ranks = self.compute_rank(truncate_num=None)
        if fix_rank:
            self.mpo_truncate_ranks = fix_rank
        else:
            self.mpo_truncate_ranks = self.compute_rank(truncate_num=self.truncate_num) # 以前是没有用到self，那么无法通过外界设置而更改

    def compute_rank_position(self, s, truncate_num=None):

        """
        Calculate the rank position in MPO bond dimension
        :param s: target bond ,type = int, range in [1:len(mpo_input_shape-1)], r_0 = r_n = 1.
        :return:  target bond 's' real bond dimension.
        """
        rank_left = 1  # ranks_left: all the shape multiply in left of 's'.
        rank_right = 1  # ranks_right: all the shape multiply in right of 's'.
        for i in range(0, s):
            rank_left = rank_left * self.mpo_input_shape[i] * self.mpo_output_shape[i]
        for i in range(s, self.num_dim):
            rank_right = rank_right * self.mpo_input_shape[i] * self.mpo_output_shape[i]
        if truncate_num == None:
            min_rank = min(rank_left, rank_right)
        else:
            min_rank = min(int(self.truncate_num), rank_left, rank_right)
        return min_rank

    def compute_rank(self, truncate_num):
        """
        :param mpo_input_shape: the input mpo shape, type = list. [i0,i1,i2,...,i_(n-1)]
        :param truncate_num: the truncate number of mpo, type = int.
        :return:max bond dimension in every bond position, type = list, [r0,r1,r2,...,r_n],r0=r_n=1
        """
        bond_dims = [1 for i in range(self.num_dim + 1)]
        for i in range(1, self.num_dim):
            bond_dims[i] = self.compute_rank_position(i, truncate_num)
        return bond_dims

    def get_tensor_set(self, inp_matrix):
        """
        Calculate the left canonical of input matrix with a given mpo_input_shape
        :param inp_matrix: the input matrix
        :param mpo_input_shape:
        :return: a tensor with left canonical in input matrix
        """
        tensor_set = []
        res = inp_matrix
        #################################################################################
        # make M(m1,m2,...,mk, n1,n2,...,nk) to M(m1,n1,m2,n2,...,mk,nk)
        res = res.reshape(tuple(self.mpo_input_shape[:]) + tuple(self.mpo_output_shape[:]))
        self.index_permute = np.transpose(
            np.array(range(len(self.mpo_input_shape) + len(self.mpo_output_shape))).reshape((2, -1))).flatten()
        res = np.transpose(res, self.index_permute)
        #################################################################################
        for i in range(self.num_dim - 1):
            # Do the SVD operator
            res = res.reshape([self.mpo_ranks[i] * self.mpo_input_shape[i] * self.mpo_output_shape[i], -1])
            u, lamda, v = svd_module.computeSVD(res)

            u = u.reshape([self.mpo_ranks[i], self.mpo_input_shape[i], self.mpo_output_shape[i], self.mpo_ranks[i+1]])
            tensor_set.append(u)
            res = np.dot(np.diag(lamda), v)
        res = res.reshape([self.mpo_ranks[self.num_dim-1], self.mpo_input_shape[self.num_dim-1],
                           self.mpo_output_shape[self.num_dim-1], self.mpo_ranks[self.num_dim]])
        tensor_set.append(res)
        return tensor_set
    def left_canonical(self,tensor_set):
        left_canonical_tensor = [0 for i in range(self.num_dim + 1)]
        mat = tensor_set[0]
        mat = mat.reshape(-1, mat.shape[3])
        u, lamda, v = svd_module.computeSVD(mat)


        left_canonical_tensor[1] = np.dot(np.diag(lamda), v)
        for i in range(1,self.num_dim-1):
            mat = np.tensordot(left_canonical_tensor[i], tensor_set[i],[1,0])
            mat = mat.reshape(-1, mat.shape[-1])
            u, lamda, v = svd_module.computeSVD(mat)



            left_canonical_tensor[i+1] = np.dot(np.diag(lamda), v)
        return left_canonical_tensor

    def right_canonical(self, tensor_set):
        """
        Calculate the right tensor canonical for MPO format required
        :param left_tensor: the tensor_set output from function: left_canonical
        :return: the right_tensor_canonical format for calculate the mpo decomposition
        """
        right_canonical_tensor = [0 for i in range(self.num_dim + 1)]
        # print(tensor_set.shape)
        mat = tensor_set[self.num_dim - 1]
        mat = mat.reshape(mat.shape[0], -1)
        u, lamda, v = svd_module.computeSVD(mat)


        right_canonical_tensor[self.num_dim - 1] = np.dot(u, np.diag(lamda))

        for i in range(self.num_dim - 2, 0, -1):
            mat = np.tensordot(tensor_set[i], right_canonical_tensor[i + 1], [3, 0])
            mat = mat.reshape(mat.shape[0], -1)
            u, lamda, v = svd_module.computeSVD(mat)
            right_canonical_tensor[i] = np.dot(u, np.diag(lamda))
        return right_canonical_tensor

    def expectrum_normalization(self, lamda):
        """
        Do the lamda normalization for calculate the needed rank for MPO structure
        :param lamda: lamda parameter from left canonical
        :return:
        """
        norm_para = np.sum(lamda ** 2) ** (0.5)
        lamda_n = lamda / norm_para
        # lamda_12 = lamda ** (-0.5)
        lamda_12 = (lamda + 1e-6) ** (-0.5)
        return lamda_n, np.diag(lamda_12)

    def gauge_aux_p_q(self, left_canonical_tensor, right_canonical_tensor):
        p = [0 for i in range(self.num_dim + 1)]
        q = [0 for i in range(self.num_dim + 1)]
        lamda_set = [0 for i in range(self.num_dim + 1)]
        lamda_set_value = [0 for i in range(self.num_dim + 1)]
        lamda_set[0] = np.ones([1,1])
        lamda_set[-1] = np.ones([1,1])
        for i in range(1, self.num_dim):
            mat = np.dot(left_canonical_tensor[i],right_canonical_tensor[i])
            u, lamda, v = svd_module.computeSVD(mat)


            lamda_n, lamda_l2 = self.expectrum_normalization(lamda)
            lamda_set[i] = lamda_n
            lamda_set_value[i] = lamda
            p[i] = np.dot(right_canonical_tensor[i], v.T)
            p[i] = np.dot(p[i],lamda_l2)
            q[i] = np.dot(lamda_l2,u.T)
            q[i] = np.dot(q[i], left_canonical_tensor[i])
        return p, q, lamda_set, lamda_set_value

    def mpo_canonical(self, tensor_set, p, q):
        tensor_set[0] = np.tensordot(tensor_set[0], p[1], [3,0])
        tensor_set[-1] = np.tensordot(q[self.num_dim-1], tensor_set[-1], [1,0])
        for i in range(1, self.num_dim-1):
            tensor_set[i] = np.tensordot(q[i],tensor_set[i],[1,0])
            tensor_set[i] = np.tensordot(tensor_set[i],p[i+1], [3,0])
        return tensor_set


    def truncated_tensor(self, tensor_set, step_train=False):
        """
        Get a untruncated tensor by mpo
        :param tensor_set: the input weight
        :return: a untruncated tensor_set by mpo
        """    
        if step_train:
            tensor_set_tmp = [i.detach().cpu().numpy() for i in tensor_set]
            cano_tensor_set = self.bi_canonical(tensor_set_tmp)
            tensor_set = torch.nn.ParameterList(
            [nn.Parameter(torch.from_numpy(i).cuda(), requires_grad=True) for i in cano_tensor_set])
            tensor_set[2].requires_grad = False

        mpo_trunc = self.mpo_truncate_ranks[:]
        for i in range(self.num_dim):
            if step_train:
                mask_noise = torch.ones_like(tensor_set[i])
            t = tensor_set[i]
            r_l = mpo_trunc[i]
            r_r = mpo_trunc[i + 1]
            if isinstance(tensor_set[i], nn.parameter.Parameter):
                if step_train:
                    mask_noise[r_l:, :, :, :] = 0.0
                    mask_noise[:r_l, :, :, r_r:] = 0.0
                    tensor_set[i].data = tensor_set[i].data * mask_noise
                else:
                    tensor_set[i].data = t[:r_l, :, :, :r_r]
            else:
                tensor_set[i] = t[:r_l, :, :, :r_r]
                assert "Check! tensor_set is not nn.parameter.Parameter"
        return tensor_set

    def matrix2mpo(self, inp_matrix, cutoff=True):
        """
        Utilize the matrix to mpo format with or without cutoff
        :param inp_matrix: the input matrix, type=list
        :param cutoff: weather cut of not, type = bool
        :return: the truncated of not mps format of input matrix
        """
        tensor_set = self.get_tensor_set(inp_matrix)
        left_canonical_tensor = self.left_canonical(tensor_set)
        right_canonical_tensor = self.right_canonical(tensor_set)
        p,q,lamda_set, lamda_set_value = self.gauge_aux_p_q(left_canonical_tensor,right_canonical_tensor)
        tensor_set = self.mpo_canonical(tensor_set,p,q)
        if cutoff != False:
            tensor_set = self.truncated_tensor(tensor_set)
        return tensor_set,lamda_set, lamda_set_value
    def bi_canonical(self, tensor_set):
        left_canonical_tensor = self.left_canonical(tensor_set)
        right_canonical_tensor = self.right_canonical(tensor_set)
        p,q,_, _ = self.gauge_aux_p_q(left_canonical_tensor,right_canonical_tensor)
        tensor_set = self.mpo_canonical(tensor_set,p,q)

        return tensor_set
    def mpo2matrix(self, tensor_set):
        """
        shirnk the bond dimension to tranfer an mpo format to matrix format
        :param tensor_set: the input mpo format
        :return: the matrix format
        """
        t = tensor_set[0]
        for i in range(1, self.num_dim):
            t = torch.tensordot(t, tensor_set[i], ([len(t.shape)-1],[0]))
        t = t.squeeze(0)
        t = t.squeeze(-1)
        tmp1 = torch.tensor(range(len(self.mpo_output_shape))) * 2
        tmp2 = tmp1 + 1
        new_index = torch.cat((tmp1, tmp2), 0)
        t = t.permute(tuple(new_index))
        t = t.reshape(torch.prod(torch.tensor(self.mpo_input_shape)),torch.prod(torch.tensor(self.mpo_output_shape)))
        return t
    def mpo2matrix_n2(self, tensor_set, new_shape=None):
        """
        shirnk the bond dimension to tranfer an mpo format to matrix format
        :param tensor_set: the input mpo format
        :return: the matrix format
        """

        t = torch.matmul(tensor_set[0], tensor_set[1]).T

        return t
    def mpo2matrix_int(self, ts1, scale1, ts2, scale2, new_shape):
        """
        shirnk the bond dimension to tranfer an mpo format to matrix format
        :param tensor_set: the input mpo format
        :return: the matrix format
        """
        ts_o = torch.matmul(ts1, ts2) # [4096,44]
        ts_o = ts_o * (torch.matmul(scale1.expand(4096, 44), scale2.expand(44,44))) / 4096.

        return ts_o.T.reshape(new_shape)
    def calculate_total_mpo_param(self, cutoff=True):
        total_size = 0
        if cutoff:
            rank = self.mpo_truncate_ranks
        else:
            rank = self.mpo_ranks
        for i in range(len(self.mpo_input_shape)):
            total_size += rank[i] * self.mpo_input_shape[i] * self.mpo_output_shape[i] * rank[i + 1]

        return total_size
    def new_mpo2matrix(self, tensor_set):
        """
        shirnk the bond dimension to tranfer an mpo format to matrix format
        :param tensor_set: the input mpo format
        :return: the matrix format
        """
        t = tensor_set[0]
        for i in range(1, self.num_dim):
            t = torch.tensordot(t, tensor_set[i], ([len(t.shape)-1],[0]))
        t = t.reshape(torch.prod(torch.tensor(self.mpo_input_shape)),torch.prod(torch.tensor(self.mpo_output_shape)))
        return t
    @staticmethod
    def test_difference(matrix1, matrix2):
        """
        we input an matrix , return the difference between those two matrix
        :param matrix:
        :return:
        """
        v = matrix1 - matrix2
        error = np.linalg.norm(v)
        return error
