#!/usr/bin/python
# -*- coding: utf-8 -*-
#
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import utils.network_utils as net_utils
from models.graph_agg import *
from config import cfg


LEAKY_VALUE = cfg.NETWORK.LEAKY_VALUE
act = nn.LeakyReLU(LEAKY_VALUE,inplace=True)

# out_shape = (H-1)//stride + 1 # for dilation=1
def conv(in_channels, out_channels, kernel_size=3, act=True, stride=1, groups=1, bias=True):
    m = []
    m.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
        padding=(kernel_size-1)//2, groups=groups, bias=bias))
    if act: m.append(nn.LeakyReLU(LEAKY_VALUE,inplace=True))
    return nn.Sequential(*m)

# out_shape = H*stride + kernel - 2*padding - stride + out_padding # for dilation=1
def upconv(in_channels, out_channels, stride=2, act=True, groups=1, bias=True):
    m = []
    kernel_size = 2 + stride
    m.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
        padding=1, groups=groups, bias=bias))
    if act: m.append(nn.LeakyReLU(LEAKY_VALUE,inplace=True))
    return nn.Sequential(*m)


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, res_scale=1, bias=True):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size-1)//2,bias=bias))
            if i == 0:
                m.append(nn.LeakyReLU(LEAKY_VALUE,inplace=True))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range=1., rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1., 1., 1.), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

def upsampler(in_channels, kernel_size=3, scale=2, act=False, bias=True):
    m = []
    if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
        for _ in range(int(math.log(scale, 2))):
            m.append(nn.Conv2d(in_channels, 4 * in_channels, kernel_size, padding=1,bias=bias))
            m.append(nn.PixelShuffle(2))
            if act: m.append(nn.LeakyReLU(LEAKY_VALUE,inplace=True))

    elif scale == 3:
        m.append(nn.Conv2d(in_channels, 9 * in_channels, kernel_size, padding=1, bias=bias))
        m.append(nn.PixelShuffle(3))
        if act: m.append(nn.LeakyReLU(LEAKY_VALUE,inplace=True))
    else:
        raise NotImplementedError
    
    return nn.Sequential(*m)



class PixelShuffle_Down(nn.Module):
    def __init__(self, scale=2):
        super(PixelShuffle_Down, self).__init__()
        self.scale = scale
    def forward(self, x):
        # assert h%scale==0 and w%scale==0
        b,c,h,w = x.size()
        x = x[:,:,:int(h-h%self.scale), :int(w-w%self.scale)] 
        out_c = c*(self.scale**2)
        out_h = h//self.scale
        out_w = w//self.scale
        out = x.contiguous().view(b, c, out_h, self.scale, out_w, self.scale)
        return out.permute(0,1,3,5,2,4).contiguous().view(b, out_c, out_h, out_w)

# ----------GCNBlock---------- #
class Graph(nn.Module):
    r"""
    Graph Construction
    """
    def __init__(self, scale, k=5, patchsize=3, stride=1, window_size=20, in_channels=256, embedcnn=None):
        r"""
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param window_size: size of matching window around each patch,
            i.e. the window_size x window_size patches around a query patch
            are used for matching
        :param in_channels: number of input channels
        :param embedcnn_opt: options for the embedding cnn
        """
        super(Graph, self).__init__()
        self.scale = scale
        self.k = k
        self.vgg = embedcnn is not None

        if embedcnn is None:
            embed_ch = 64
            embed_out = 8
            self.embedcnn = nn.Sequential(
                conv(in_channels, embed_ch, kernel_size=3),
                conv(embed_ch, embed_ch, kernel_size=3),
                conv(embed_ch, embed_out, kernel_size=3)
            )
        else:
            self.embedcnn = embedcnn

        indexer = lambda xe_patch,ye_patch: index_neighbours(xe_patch, ye_patch, window_size, scale)

        self.graph_construct = GraphConstruct(scale=scale, indexer=indexer, k=k,
                patchsize=patchsize, stride=stride)


    def forward(self, x, y):
        # x: son features, y: father features

        xe = self.embedcnn(x)
        ye = self.embedcnn(y)

        score_k, idx_k, diff_patch = self.graph_construct(xe, ye)
        return score_k, idx_k, diff_patch

class GCNBlock(nn.Module):
    r"""
    Graph Aggregation
    """
    def __init__(self, nplanes_in, scale, k=5, patchsize=3, stride=1, diff_n=64):
        r"""
        :param nplanes_in: number of input features
        :param scale: downsampling factor
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param diff_n: number of diff vector channels
        """        
        super(GCNBlock, self).__init__()
        self.nplanes_in = nplanes_in
        self.scale = scale
        self.k = k

        self.pixelshuffle_down = PixelShuffle_Down(scale)

        self.graph_aggregate = GraphAggregation(scale=scale, k=k,
                patchsize=patchsize, stride=stride)

        self.knn_downsample = nn.Sequential(
                conv(nplanes_in, nplanes_in, kernel_size=5, stride=scale),
                conv(nplanes_in, nplanes_in, kernel_size=3),
                conv(nplanes_in, nplanes_in, kernel_size=3, act=False)
            )

        self.diff_downsample = nn.AvgPool2d(kernel_size=scale, stride=scale)

        self.weightnet_lr = nn.Sequential(
                        conv(diff_n, 64, kernel_size=1, act=False),
                        ResBlock(64, kernel_size=1, res_scale=cfg.NETWORK.RES_SCALE),
                        conv(64, 1, kernel_size=1, act=False)
                    )
        self.weightnet_hr = nn.Sequential(
                        conv(diff_n, 64, kernel_size=1, act=False),
                        ResBlock(64, kernel_size=1, res_scale=cfg.NETWORK.RES_SCALE),
                        conv(64, 1, kernel_size=1, act=False)
                    )

 
    def weight_edge(self, knn_hr, diff_patch):
        b, c, h_hr, w_hr  = knn_hr.shape
        b, ce, _, _ = diff_patch.shape
        h_lr, w_lr = h_hr//self.scale, w_hr//self.scale

        knn_hr = knn_hr.view(b, self.k, c//self.k, h_hr, w_hr)
        diff_patch = diff_patch.view(b, self.k, ce//self.k, h_hr, w_hr)

        knn_lr, weight_lr, weight_hr = [],[],[]
        for i in range(self.k):
            knn_lr.append(self.knn_downsample(knn_hr[:,i]).view(b, 1, c//self.k, h_lr, w_lr))
            diff_patch_lr = self.diff_downsample(diff_patch[:,i])
            weight_lr.append(self.weightnet_lr(diff_patch_lr))
            weight_hr.append(self.weightnet_hr(diff_patch[:,i]))


        weight_lr = torch.cat(weight_lr, dim=1)
        weight_lr = weight_lr.view(b, self.k, 1, h_lr, w_lr)
        weight_lr = F.softmax(weight_lr, dim=1)
     
        weight_hr = torch.cat(weight_hr, dim=1)
        weight_hr = weight_hr.view(b, self.k, 1, h_hr, w_hr)
        weight_hr = F.softmax(weight_hr, dim=1)

        knn_lr = torch.cat(knn_lr, dim=1)
        knn_lr = torch.sum(knn_lr*weight_lr, dim=1, keepdim=False)
        knn_hr = torch.sum(knn_hr*weight_hr, dim=1, keepdim=False)

        return knn_lr, knn_hr


    def forward(self, y, idx_k, diff_patch):

        # graph_aggregate
        yd = self.pixelshuffle_down(y)

        # b k*c h*s w*s
        knn_hr = self.graph_aggregate(y, yd, idx_k)

        # for diff socre
        knn_lr, knn_hr = self.weight_edge(knn_hr, diff_patch)

        return knn_lr, knn_hr
