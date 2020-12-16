#!/usr/bin/pythonupsampler
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

from models.submodules import *
from models.VGG19 import VGG19
from config import cfg

class IGNN(nn.Module):
    def __init__(self):
        super(IGNN, self).__init__()

        kernel_size = 3 
        n_resblocks = cfg.NETWORK.N_RESBLOCK
        n_feats = cfg.NETWORK.N_FEATURE
        n_neighbors = cfg.NETWORK.N_REIGHBOR
        scale = cfg.CONST.SCALE
        if cfg.CONST.SCALE == 4:
            scale = 2
        window = cfg.NETWORK.WINDOW_SIZE
        gcn_stride = 2
        patch_size = 3

        self.sub_mean = MeanShift(rgb_range=cfg.DATA.RANGE, sign=-1)
        self.add_mean = MeanShift(rgb_range=cfg.DATA.RANGE, sign=1)

        self.vggnet = VGG19([3])

        self.graph = Graph(scale, k=n_neighbors, patchsize=patch_size, stride=gcn_stride, 
            window_size=window, in_channels=256, embedcnn=self.vggnet)

        # define head module
        self.head = conv(3, n_feats, kernel_size, act=False)
        # middle 16
        pre_blocks = int(n_resblocks//2)
        # define body module
        m_body1 = [
            ResBlock(
                n_feats, kernel_size, res_scale=cfg.NETWORK.RES_SCALE 
            ) for _ in range(pre_blocks)
        ]

        m_body2 = [
            ResBlock(
                n_feats, kernel_size, res_scale=cfg.NETWORK.RES_SCALE 
            ) for _ in range(n_resblocks-pre_blocks)
        ]

        m_body2.append(conv(n_feats, n_feats, kernel_size, act=False))

        fuse_b = [
            conv(n_feats*2, n_feats, kernel_size),
            conv(n_feats, n_feats, kernel_size, act=False) # act=False important for relu!!!
        ]

        fuse_up = [
            conv(n_feats*2, n_feats, kernel_size),
            conv(n_feats, n_feats, kernel_size)        
        ]

        if cfg.CONST.SCALE == 4:
            m_tail = [
                upsampler(n_feats, kernel_size, scale, act=False),
                conv(n_feats, 3, kernel_size, act=False)  # act=False important for relu!!!
            ]
        else:
            m_tail = [
                conv(n_feats, 3, kernel_size, act=False)  # act=False important for relu!!!
            ]            

        self.body1 = nn.Sequential(*m_body1)
        self.gcn = GCNBlock(n_feats, scale, k=n_neighbors, patchsize=patch_size, stride=gcn_stride)

        self.fuse_b = nn.Sequential(*fuse_b)

        self.body2 = nn.Sequential(*m_body2)
       
        self.upsample = upsampler(n_feats, kernel_size, scale, act=False)
        self.fuse_up = nn.Sequential(*fuse_up)

        self.tail = nn.Sequential(*m_tail)


    def forward(self, x_son, x):

        score_k, idx_k, diff_patch = self.graph(x_son, x)
        idx_k = idx_k.detach()
        if cfg.NETWORK.WITH_DIFF:
            diff_patch = diff_patch.detach()

        x = self.sub_mean(x)
        x0 = self.head(x)
        x1 = self.body1(x0)
        x1_lr, x1_hr = self.gcn(x1, idx_k, diff_patch)
        x1 = self.fuse_b(torch.cat([x1, x1_lr], dim=1)) 
        x2 = self.body2(x1) + x0
        x = self.upsample(x2)
        x = self.fuse_up(torch.cat([x, x1_hr], dim=1))
        x= self.tail(x)
        x = self.add_mean(x)

        return x 
