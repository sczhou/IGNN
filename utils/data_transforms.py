#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
'''ref: http://pytorch.org/docs/master/torchvision/transforms.html'''


import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from config import cfg
from PIL import Image
import random
import numbers
class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms[:-1]
        self.ToTensor = transforms[-1]

    def __call__(self, img_lr, img_hr):
        for t in self.transforms:
            img_lr, img_hr = t(img_lr, img_hr)

        img_lr, img_hr = self.ToTensor(img_lr, img_hr)

        return img_lr, img_hr


class ColorJitter(object):
    def __init__(self, color_adjust_para):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = color_adjust_para

    def __call__(self, img_lr, img_hr):
        if random.random() > 0.8:
            return img_lr, img_hr
        img_lr, img_hr = [Image.fromarray(np.uint8(img)) for img in [img_lr, img_hr]]
        if self.brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            img_lr, img_hr = [F.adjust_brightness(img, brightness_factor) for img in [img_lr, img_hr]]

        if self.contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            img_lr, img_hr = [F.adjust_contrast(img, contrast_factor) for img in [img_lr, img_hr]]

        if self.saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            img_lr, img_hr = [F.adjust_saturation(img, saturation_factor) for img in [img_lr, img_hr]]

        if self.hue > 0:
            hue_factor = np.random.uniform(-self.hue, self.hue)
            img_lr, img_hr = [F.adjust_hue(img, hue_factor) for img in [img_lr, img_hr]]

        img_lr, img_hr = [np.asarray(img) for img in [img_lr, img_hr]]
        img_lr, img_hr = [img.clip(0,255) for img in [img_lr, img_hr]]

        return img_lr, img_hr

class RandomColorChannel(object):
    def __call__(self, img_lr, img_hr):
        if random.random() > 0.8:
            return img_lr, img_hr
        random_order = np.random.permutation(3)
        img_lr, img_hr = [img[:,:,random_order] for img in [img_lr, img_hr]]

        return img_lr, img_hr

class BGR2RGB(object):
    def __call__(self, img_lr, img_hr):
        random_order = [2,1,0]
        img_lr, img_hr = [img[:,:,random_order] for img in [img_lr, img_hr]]

        return img_lr, img_hr

class RandomGaussianNoise(object):
    def __init__(self, gaussian_para):
        self.sigma = gaussian_para

    def __call__(self, img_lr, img_hr):
        if random.random() > 0.8:
            return img_lr, img_hr
        noise_std = np.random.randint(1, self.sigma)

        gaussian_noise = np.random.randn(*img_lr.shape)*noise_std
        # only apply to lr images
        img_lr = (img_lr + gaussian_noise).clip(0, 255)

        return img_lr, img_hr

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
    def __call__(self, img_lr, img_hr):
        assert(all([isinstance(img, np.ndarray) for img in [img_lr, img_hr]]))
        img_lr, img_hr = [img/self.std -self.mean for img in [img_lr, img_hr]]

        return img_lr, img_hr


class RandomCrop(object):

    def __init__(self, crop_size, scale):
        self.scale = scale
        self.crop_h_hr, self.crop_w_hr  = crop_size
        assert(np.all(np.array(crop_size) % scale == 0))
        self.crop_h_lr, self.crop_w_lr  = np.array(crop_size)//scale

    def __call__(self, img_lr, img_hr):
        lr_size_h, lr_size_w, _ = img_lr.shape
        rnd_h_lr  = random.randint(0, lr_size_h - self.crop_h_lr)
        rnd_w_lr  = random.randint(0, lr_size_w - self.crop_w_lr)
        rnd_h_hr, rnd_w_hr = int(rnd_h_lr * self.scale), int(rnd_w_lr * self.scale)

        img_lr = img_lr[rnd_h_lr:rnd_h_lr + self.crop_h_lr, rnd_w_lr:rnd_w_lr + self.crop_w_lr, :]
        img_hr = img_hr[rnd_h_hr:rnd_h_hr + self.crop_h_hr, rnd_w_hr:rnd_w_hr + self.crop_w_hr, :]

        return img_lr, img_hr

class BorderCrop(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img_lr, img_hr):
        ih, iw = img_lr.shape[:2]
        img_hr = img_hr[0:ih * self.scale, 0:iw * self.scale]
        return img_lr, img_hr

class FlipRotate(object):

    def __call__(self, img_lr, img_hr):

        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        if hflip: img_lr, img_hr = [np.fliplr(img) for img in [img_lr, img_hr]]
        if vflip: img_lr, img_hr = [np.flipud(img) for img in [img_lr, img_hr]]
        if rot90: img_lr, img_hr = [img.transpose(1, 0, 2) for img in [img_lr, img_hr]]

        img_lr, img_hr = [img.clip(0,255) for img in [img_lr, img_hr]]

        return img_lr, img_hr

class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, img_lr, img_hr):
        assert(all([isinstance(img, np.ndarray) for img in [img_lr, img_hr]]))
        img_lr, img_hr = [img.astype('float32') for img in [img_lr, img_hr]]

        img_lr, img_hr = [np.transpose(img, (2, 0, 1)) for img in [img_lr, img_hr]]
        img_lr, img_hr = [np.ascontiguousarray(img) for img in [img_lr, img_hr]]
        img_lr, img_hr = [torch.from_numpy(img) for img in [img_lr, img_hr]]

        img_lr, img_hr = [img.clamp(0,255) for img in [img_lr, img_hr]]

        return img_lr, img_hr


