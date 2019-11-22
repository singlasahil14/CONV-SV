import argparse
import os
import shutil
import time
import math

import torch
from torch.autograd import gradcheck
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
 
def power_iteration(W, u=None, v=None, num_iters=50, return_vectors=False):
    if u is None:
        u = torch.randn((1, W.shape[1]), device='cuda')
        u_norm = torch.norm(u, dim=1)
        u_n = u/u_norm
    else:
        u_n = u
        v_n = v
    for i in range(num_iters):
        v = torch.matmul(u_n, W.t())
        v_norm = torch.norm(v, dim=1)
        v_n = v/v_norm

        u = torch.matmul(v_n, W)
        u_norm = torch.norm(u, dim=1)
        u_n = u/u_norm
    sigma = (v_n.mm(W)).mm(u_n.t())
    if return_vectors:
        return sigma[0, 0], u_n, v_n
    return sigma[0, 0]

class ConvFilterNorm(nn.Module):
    def __init__(self, conv_filter):
        super(ConvFilterNorm, self).__init__()
        out_ch, in_ch, h, w = conv_filter.shape
        conv_filter_permute = conv_filter.permute(dims=(0, 2, 1, 3))
        conv_filter_matrix = conv_filter_permute.contiguous().view(out_ch*h, -1)

        self.sigma, u, v = power_iteration(conv_filter_matrix, num_iters=50, return_vectors=True)
        self.u = u.detach()
        self.v = v.detach()

    def forward(self, conv_filter):
        out_ch, in_ch, h, w = conv_filter.shape
        conv_filter_permute = conv_filter.permute(dims=(0, 2, 1, 3))
        conv_filter_matrix = conv_filter_permute.contiguous().view(out_ch*h, -1)

        _, u, v = power_iteration(conv_filter_matrix, self.u, self.v, num_iters=10, return_vectors=True)
        self.u = u.detach()
        self.v = v.detach()
        return math.sqrt(h*w)*MatrixNormFunction.apply(conv_filter_matrix, self.u, self.v)

class MatrixNorm(nn.Module):
    def __init__(self, matrix):
        super(MatrixNorm, self).__init__()
        self.sigma, u, v = power_iteration(matrix, num_iters=50, return_vectors=True)
        self.u = u.detach()
        self.v = v.detach()

    def forward(self, matrix):
        _, u, v = power_iteration(matrix, self.u, self.v, num_iters=10, return_vectors=True)
        self.u = u.detach()
        self.v = v.detach()
        return MatrixNormFunction.apply(matrix, self.u, self.v)

class MatrixNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, u, v):
        sigma = (v.mm(matrix)).mm(u.t())
        ctx.save_for_backward(matrix, u, v)
        return sigma

    @staticmethod
    def backward(ctx, grad_output):
        filter_matrix, u, v = ctx.saved_tensors
        grad_weight = grad_output.clone()
        grad_singular = ((v.t()).mm(u))
        return grad_weight*grad_singular, None, None
