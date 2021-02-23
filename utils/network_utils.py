#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import os
import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import random
import numpy as np
from datetime import datetime as dt
from config import cfg
import torch.nn.functional as F

import cv2

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for n, m in net.named_modules():
            if isinstance(m, nn.Conv2d) and not '_mean' in n:
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def set_random_seed(seed, deterministic=False):
    """Set random seed just for debug.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(file_path, file_name, epoch_idx, net, solver, best_PSNR, best_epoch):
    if not os.path.exists(file_path): 
        mkdir(file_path)
    file_path = os.path.join(file_path, file_name)
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'best_PSNR': best_PSNR,
        'best_epoch': best_epoch,
        'net_state_dict': net.state_dict(),
        'solver_state_dict': solver.state_dict()
    }
    torch.save(checkpoint, file_path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_weight_parameters(model):
    return [param for name, param in model.named_parameters() if ('weight' in name)]

def get_bias_parameters(model):
    return [param for name, param in model.named_parameters() if ('bias' in name)]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.5f} ({:.5f})'.format(self.val, self.avg)


##################################
# Image convert functions
##################################
def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)].contiguous()

def crop_border(img_list, crop_border):
    """Crop borders of images
    Args:
        img_list (list [Numpy]): HWC
        crop_border (int): crop border for each end of height and weight
    Returns:
        (list [Numpy]): cropped image (list)
    """
    if crop_border == 0: return img_list

    if isinstance(img_list, list):
        return [v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list]
    else:
        return v[crop_border:-crop_border, crop_border:-crop_border]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 255)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    assert tensor.dim() == 3

    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))[:,:,[2,1,0]]  # HWC, to BGR for cv2 imwrite
    
    if out_type == np.uint8:
        img_np = img_np.round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def img2tensor(img, min_max=(0, 255)):
    '''
    Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    '''
    img = img.astype('float32')
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    return img.clamp(*min_max)

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    For matplotlib.image
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    For opencv
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

##################################
# adaptive_instance_normalization
##################################

def adaptive_instance_normalization(center_feat, knn_feat, eps=1e-5):
    # center_feat = center_feat.contiguous()
    # knn_feat = knn_feat.contiguous()
    b,m,c,p = center_feat.size()
    _,_,_,sp,k = knn_feat.size()

    center_var = center_feat.var(dim=3) + eps
    center_std = center_var.sqrt().view(b, m, c, 1, 1)
    center_mean = center_feat.mean(dim=3).view(b, m, c, 1, 1)

    knn_var = knn_feat.var(dim=3) + eps
    knn_std = knn_var.sqrt().view(b, m, c, 1, k)
    knn_mean = knn_feat.mean(dim=3).view(b, m, c, 1, k)

    size = knn_feat.size()
    normalized_feat = (knn_feat - knn_mean.expand(size)) / knn_std.expand(size)

    return normalized_feat * center_std.expand(size) + center_mean.expand(size)


##################################
# Efficient and effective test
##################################
class Chop(nn.Module):
    def __init__(self, model):
        super(Chop, self).__init__()
        self.model = model
        self.scale = cfg.CONST.SCALE
        if cfg.CONST.SCALE == 4:      
            self.scale_s = 2
        else:
            self.scale_s = cfg.CONST.SCALE

    def forward(self, x_son, x, shave=12, min_size=40000):
        n_GPUs = min(cfg.CONST.NUM_GPU, 4)
        b, c, h_s, w_s = x_son.size()
        _, _, h, w = x.size()
        h_0, w_0 = h%self.scale_s, w%self.scale_s

        h_half_s, w_half_s = h_s//2, w_s//2
        h_half, w_half = h_half_s*self.scale_s, w_half_s*self.scale_s

        shave_s = shave//self.scale_s

        h_size_s, w_size_s = h_half_s + shave_s, w_half_s + shave_s
        h_size, w_size = h_size_s*self.scale_s, w_size_s*self.scale_s

        lr_list_s = [
            x_son[:, :, 0:h_size_s, 0:w_size_s],
            x_son[:, :, 0:h_size_s, (w_s - w_size_s):w_s],
            x_son[:, :, (h_s - h_size_s):h_s, 0:w_size_s],
            x_son[:, :, (h_s - h_size_s):h_s, (w_s - w_size_s):w_s]]

        lr_list = [
            x[:, :, 0:h_size+h_0, 0:w_size+w_0],
            x[:, :, 0:h_size+h_0, (w - w_size-w_0):w],
            x[:, :, (h - h_size-h_0):h, 0:w_size+w_0],
            x[:, :, (h - h_size-h_0):h, (w - w_size-w_0):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_s_batch = torch.cat(lr_list_s[i:(i + n_GPUs)], dim=0)
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_s_batch, lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward(patch_s, patch, shave=shave, min_size=min_size) \
                for (patch_s, patch) in zip(lr_list_s, lr_list)
            ]

        h, w = self.scale * h, self.scale * w

        h_half, w_half = self.scale * h_half, self.scale * w_half
        h_size, w_size = self.scale * h_size, self.scale * w_size
        shave *= self.scale
        h_0, w_0 = h_0*self.scale, w_0*self.scale
        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half+w_0):w_size+w_0]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half+h_0):h_size+h_0, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half+h_0):h_size+h_0, (w_size - w + w_half+w_0):w_size+w_0]

        return output


def forward_x8(x_son, x, forward_function):
    def _transform(v, op):

        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).cuda()

        return ret

    lr_son_list = [x_son]
    lr_list = [x]
    for tf in 'v', 'h', 't':
        lr_son_list.extend([_transform(t, tf) for t in lr_son_list])
        lr_list.extend([_transform(t, tf) for t in lr_list])

    sr_list = [forward_function(lr_son, lr) for (lr_son, lr) in zip(lr_son_list, lr_list)]
    for i in range(len(sr_list)):
        if i > 3:
            sr_list[i] = _transform(sr_list[i], 't')
        if i % 4 > 1:
            sr_list[i] = _transform(sr_list[i], 'h')
        if (i % 4) % 2 == 1:
            sr_list[i] = _transform(sr_list[i], 'v')

    output_cat = torch.cat(sr_list, dim=0)
    output = output_cat.mean(dim=0, keepdim=True)

    return output
