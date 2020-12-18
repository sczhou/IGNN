'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

from collections import namedtuple
import pyinn
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matmul_cuda

from torch.autograd import Variable


class IndexedMatmul1Efficient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, I):
        ctx.save_for_backward(x, y, I)
        b = y.shape[0]
        m = y.shape[1]
        n = x.shape[1]
        o = I.shape[2]
        e = x.shape[2]
        out = torch.tensor(np.zeros(b*m*o), dtype=torch.float).reshape(b,m,o).cuda()
        matmul_cuda.matmul1(x,y,I,out,n,m,e,o,b)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, y, I = ctx.saved_tensors
        b = y.shape[0]
        m = y.shape[1]
        n = x.shape[1]
        o = I.shape[2]
        e = x.shape[2]
        grad_x = torch.tensor(np.zeros(x.numel()), dtype=torch.float).reshape(x.shape[0],x.shape[1],x.shape[2]).cuda()
        grad_y = torch.tensor(np.zeros(y.numel()), dtype=torch.float).reshape(y.shape[0],y.shape[1],y.shape[2]).cuda()
        matmul_cuda.matmul1_bwd(grad,x,y,I,grad_x,grad_y, m, n, e, o, b)
        return grad_x, grad_y, I


class IndexedMatmul2Efficient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, I, chunk_size=256):
        ctx.save_for_backward(x, y, I)
        ctx.chunk_size = chunk_size
        b,_,o,k = y.shape
        n,e = x.shape[1:3]
        m = I.shape[1]
        x_interm = x.view(b,1,n,e).detach()
        z_chunks = []
        for m_offset in range(0,m,chunk_size):
            this_chunk_size = min(chunk_size, m-m_offset)
            I_chunk = I[:,m_offset:m_offset+this_chunk_size,:]
            y_chunk = y[:,m_offset:m_offset+this_chunk_size,:,:]

            If = I_chunk.view(b,1,this_chunk_size,o).expand(b,k,this_chunk_size,o)
            y_full = torch.cuda.FloatTensor(b,k,this_chunk_size,n).fill_(0)
            y_full = y_full.scatter_add(src=y_chunk.permute(0,3,1,2), index=If, dim=3)
            z_interm = torch.cat([torch.matmul(y_full[:,i_k:i_k+1,:,:], x_interm) for i_k in range(k)], 1)
            z_chunk = z_interm.permute(0,2,3,1)
            z_chunks.append(z_chunk)
        z = torch.cat(z_chunks, 1)
        return z

    @staticmethod
    def backward(ctx, grad):
        x, y, I = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        b,_,o,k = y.shape
        n,e = x.shape[1:3]
        m = I.shape[1]
        x_interm = x.view(b,1,n,e).detach()
        grad_x = torch.zeros_like(x)
        grad_y_chunks = []

        for m_offset in range(0,m,chunk_size):
            this_chunk_size = min(chunk_size, m-m_offset)
            I_chunk = I[:,m_offset:m_offset+this_chunk_size,:]
            y_chunk = y[:,m_offset:m_offset+this_chunk_size,:,:]
            grad_chunk = grad[:,m_offset:m_offset+this_chunk_size,:,:].permute(0,3,2,1)

            If = I_chunk.view(b,1,this_chunk_size,o).expand(b,k,this_chunk_size,o)
            del I_chunk
            y_full = torch.cuda.FloatTensor(b,k,this_chunk_size,n).fill_(0)
            y_full = y_full.scatter_add(src=y_chunk.permute(0,3,1,2), index=If, dim=3)
            del y_chunk

            for i_k in range(k):
                grad_x += torch.matmul(grad_chunk[:,i_k,:,:], y_full[:,i_k,:,:]).permute(0,2,1)

            del y_full
            grad_y_full = torch.cat([torch.matmul(x_interm, grad_chunk[:,i_k:i_k+1,:,:]) for i_k in range(k)], 1)
            del grad_chunk
            grad_y_chunk = grad_y_full.gather(2, If.permute(0,1,3,2)).permute(0,3,2,1)
            del grad_y_full
            grad_y_chunks.append(grad_y_chunk)

        grad_y = torch.cat(grad_y_chunks, 1)
        return grad_x, grad_y, None, None

# class IndexedMatmul2Efficient_noidx(torch.autograd.Function):
#     # without index I
#     # x-yd: b,n,f
#     # y-W: b,m,o,k
#     @staticmethod
#     def forward(ctx, x, y, chunk_size=256):
#         ctx.save_for_backward(x, y)
#         ctx.chunk_size = chunk_size
#         b,m,o,k = y.shape
#         n,e = x.shape[1:3]
#         x_interm = x.view(b,1,n,e).detach()
#         z_chunks = []
#         for m_offset in range(0,m,chunk_size):
#             this_chunk_size = min(chunk_size, m-m_offset)
#             # I_chunk = I[:,m_offset:m_offset+this_chunk_size,:]
#             y_chunk = y[:,m_offset:m_offset+this_chunk_size,:,:]

#             # If = I_chunk.view(b,1,this_chunk_size,o).expand(b,k,this_chunk_size,o)
#             # y_full = torch.cuda.FloatTensor(b,k,this_chunk_size,n).fill_(0)
#             # y_full = y_full.scatter_add(src=y_chunk.permute(0,3,1,2), index=If, dim=3)
#             y_full = y_chunk.permute(0,3,1,2)

#             z_interm = torch.cat([torch.matmul(y_full[:,i_k:i_k+1,:,:], x_interm) for i_k in range(k)], 1)
#             z_chunk = z_interm.permute(0,2,3,1)
#             z_chunks.append(z_chunk)
#         z = torch.cat(z_chunks, 1)
#         return z

#     @staticmethod
#     def backward(ctx, grad):
#         x, y = ctx.saved_tensors
#         chunk_size = ctx.chunk_size
#         b,m,o,k = y.shape
#         n,e = x.shape[1:3]
#         x_interm = x.view(b,1,n,e).detach()
#         grad_x = torch.zeros_like(x)
#         grad_y_chunks = []

#         for m_offset in range(0,m,chunk_size):
#             this_chunk_size = min(chunk_size, m-m_offset)
#             # I_chunk = I[:,m_offset:m_offset+this_chunk_size,:]
#             y_chunk = y[:,m_offset:m_offset+this_chunk_size,:,:]
#             grad_chunk = grad[:,m_offset:m_offset+this_chunk_size,:,:].permute(0,3,2,1)

#             # If = I_chunk.view(b,1,this_chunk_size,o).expand(b,k,this_chunk_size,o)
#             # del I_chunk
#             # y_full = torch.cuda.FloatTensor(b,k,this_chunk_size,n).fill_(0)
#             # y_full = y_full.scatter_add(src=y_chunk.permute(0,3,1,2), index=If, dim=3)
#             y_full = y_chunk.permute(0,3,1,2)
#             del y_chun

#             for i_k in range(k):
#                 grad_x += torch.matmul(grad_chunk[:,i_k,:,:], y_full[:,i_k,:,:]).permute(0,2,1)

#             del y_full
#             grad_y_full = torch.cat([torch.matmul(x_interm, grad_chunk[:,i_k:i_k+1,:,:]) for i_k in range(k)], 1)
#             del grad_chunk
#             # grad_y_chunk = grad_y_full.gather(2, If.permute(0,1,3,2)).permute(0,3,2,1)
#             grad_y_chunk = grad_y_full.permute(0,3,2,1)
#             del grad_y_full
#             grad_y_chunks.append(grad_y_chunk)

#         grad_y = torch.cat(grad_y_chunks, 1)
#         return grad_x, grad_y, None, None

def indexed_matmul_1_efficient(x,y,I):
    return IndexedMatmul1Efficient.apply(x,y,I)


def indexed_matmul_2_efficient(x,y,I, chunk_size=1024):
    return IndexedMatmul2Efficient.apply(x,y,I,chunk_size)

def euclidean_distance(x,y):
    out = -2*torch.matmul(x, y)
    out += (x**2).sum(dim=-1, keepdim=True)
    out += (y**2).sum(dim=-2, keepdim=True)
    return out


def calc_padding(x_shape, patchsize, stride, padding=None):
    if padding is None:
        xdim = x_shape
        padvert = -(xdim[0] - patchsize) % stride
        padhorz = -(xdim[1] - patchsize) % stride

        padtop = int(np.floor(padvert / 2.0))
        padbottom = int(np.ceil(padvert / 2.0))
        padleft = int(np.floor(padhorz / 2.0))
        padright = int(np.ceil(padhorz / 2.0))
    else:
        padtop = padbottom = padleft = padright = padding

    return padtop, padbottom, padleft, padright


def im2patch(x, patchsize, stride, padding=None, returnpadding=False):
    padtop, padbottom, padleft, padright = calc_padding(x.shape[2:], patchsize, stride, padding)
    xpad = F.pad(x, pad=(padleft, padright, padtop, padbottom))

    x2col = pyinn.im2col(xpad, [patchsize]*2, [stride]*2, [0,0])
    if returnpadding:
        return x2col, (padtop, padbottom, padleft, padright)
    else:
        return x2col

def patch2im(x_patch, patchsize, stride, padding):
    padtop, padbottom, padleft, padright = padding
    counts = pyinn.col2im(torch.ones_like(x_patch), [patchsize]*2, [stride]*2, [0,0])
    x = pyinn.col2im(x_patch.contiguous(), [patchsize]*2, [stride]*2, [0,0])
    x = x/counts
    x = x[:,:,padtop:x.shape[2]-padbottom, padleft:x.shape[3]-padright]
    return x

class Im2Patch(nn.Module):
    def __init__(self, patchsize, stride, padding=None):
        super(Im2Patch, self).__init__()
        self.patchsize = patchsize
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return im2patch(x, self.patchsize, self.stride, self.padding)

class Patch2Im(nn.Module):
    def __init__(self, patchsize, stride, padding=None):
        super(Patch2Im, self).__init__()
        self.patchsize = patchsize
        self.stride = stride
        self.padding = padding

    def forward(self, x_patch):
        return patch2im(x_patch, self.patchsize, self.stride, self.padding)

# This follows semantics of numpy.finfo.
_Finfo = namedtuple('_Finfo', ['eps', 'tiny'])
_FINFO = {
    torch.HalfStorage: _Finfo(eps=0.00097656, tiny=6.1035e-05),
    torch.FloatStorage: _Finfo(eps=1.19209e-07, tiny=1.17549e-38),
    torch.DoubleStorage: _Finfo(eps=2.22044604925e-16, tiny=2.22507385851e-308),
    torch.cuda.HalfStorage: _Finfo(eps=0.00097656, tiny=6.1035e-05),
    torch.cuda.FloatStorage: _Finfo(eps=1.19209e-07, tiny=1.17549e-38),
    torch.cuda.DoubleStorage: _Finfo(eps=2.22044604925e-16, tiny=2.22507385851e-308),
}


def _finfo(tensor):
    r"""
    Return floating point info about a `Tensor` or `Variable`:
    - `.eps` is the smallest number that can be added to 1 without being lost.
    - `.tiny` is the smallest positive number greater than zero
      (much smaller than `.eps`).
    Args:
        tensor (Tensor or Variable): tensor or variable of floating point data.
    Returns:
        _Finfo: a `namedtuple` with fields `.eps` and `.tiny`.
    """
    if isinstance(tensor, Variable):
        return _FINFO[tensor.data.storage_type()]
    else:
        return _FINFO[tensor.storage_type()]

def clamp_probs(probs):
    eps = _finfo(probs).eps
    return probs.clamp(min=eps, max=1 - eps)