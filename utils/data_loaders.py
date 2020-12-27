#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import cv2
import json
import os
import io
import torch.utils.data.dataset
from utils.imresize import imresize
from utils.network_utils import img2tensor


from config import cfg
from datetime import datetime as dt
from enum import Enum, unique

class DatasetType(Enum):
    TRAIN = 0
    TEST  = 1

class SRDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, file_list, transforms = None):
        self.file_list  = file_list
        self.transforms = transforms
        if cfg.CONST.SCALE == 4:
            self.down_scale = 2
        else:
            self.down_scale = cfg.CONST.SCALE

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name, img_lr, img_hr = self.image_read(idx)
        img_lr, img_hr = self.transforms(img_lr, img_hr)
        _,h,w = img_lr.size()
        img_lr_ = img_lr[:,:int(h-h%self.down_scale), :int(w-w%self.down_scale)] 
        img_lr_s = imresize(img_lr_/255.0, 1.0/self.down_scale)*255

        img_lr_s = img_lr_s.clamp(0,255)
        return img_name, img_lr_s, img_lr, img_hr

    def image_read(self, idx):
        return self.file_list[idx]['img_name'], \
               cv2.imread(self.file_list[idx]['img_lr']), \
               cv2.imread(self.file_list[idx]['img_hr'])

# //////////////////////////////// = End of SRDataset Class Definition = ///////////////////////////////// #

class SRDataLoader:
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        if dataset_type == DatasetType.TRAIN: # for train dataset
            self.img_lr_path_template = cfg.DIR.IMAGE_LR_TRAIN_PATH
            self.img_hr_path_template = cfg.DIR.IMAGE_HR_TRAIN_PATH
            with io.open(cfg.DIR.DATASET_JSON_TRAIN_PATH, encoding='utf-8') as file:
                self.files_list = json.loads(file.read())
        elif dataset_type == DatasetType.TEST: # for val/test dataset
            self.img_lr_path_template = cfg.DIR.IMAGE_LR_TEST_PATH
            self.img_hr_path_template = cfg.DIR.IMAGE_HR_TEST_PATH
            with io.open(cfg.DIR.DATASET_JSON_TEST_PATH, encoding='utf-8') as file:
                self.files_list = json.loads(file.read())

    def get_dataset(self, transforms=None):
        files = []
        # Load data for each category
        for file in self.files_list:
            if self.dataset_type == DatasetType.TRAIN and file['phase'] == 'train':
                phase = file['phase']
                samples = file['sample']
                print('[INFO] %s Collecting files [phase = %s]' % (dt.now(), phase))
                files.extend(self.get_files(phase, samples))
            elif self.dataset_type == DatasetType.TEST and file['phase'] in ['valid','test']:
                phase = file['phase']
                samples = file['sample']
                print('[INFO] %s Collecting files [phase = %s]' % (dt.now(), phase))
                files.extend(self.get_files(phase, samples))

        print('[INFO] %s Complete collecting files of the dataset for %s. Total images: %d.' % (dt.now(), self.dataset_type.name, len(files)))
        return SRDataset(files, transforms)

    def get_files(self, phase, samples):
        files = []
        for sample_idx, sample_name in enumerate(samples):
            # Get file path of img
            img_lr_path = self.img_lr_path_template.format(phase,sample_name)
            img_hr_path = self.img_hr_path_template.format(phase,sample_name)
 
            if os.path.exists(img_lr_path) and os.path.exists(img_hr_path):
                files.append({
                    'img_name': sample_name,
                    'img_lr'  : img_lr_path,
                    'img_hr'  : img_hr_path
                })
        return files

# /////////////////////////////// = End of SRDataLoader Class Definition = /////////////////////////////// #

class TestDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, file_list, transforms = None):
        self.file_list  = file_list
        # self.transforms = transforms
        if cfg.CONST.SCALE == 4:
            self.down_scale = 2
        else:
            self.down_scale = cfg.CONST.SCALE

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name, img_lr = self.image_read(idx)
        img_lr = img_lr[:,:,[2,1,0]]
        img_lr = img2tensor(img_lr)
        _,h,w = img_lr.size()
        img_lr_ = img_lr[:,:int(h-h%self.down_scale), :int(w-w%self.down_scale)] 
        img_lr_s = imresize(img_lr_/255.0, 1.0/self.down_scale)*255

        img_lr_s = img_lr_s.clamp(0,255)
        return img_name, img_lr_s, img_lr

    def image_read(self, idx):
        return self.file_list[idx]['img_name'], \
               cv2.imread(self.file_list[idx]['img_lr'])

# //////////////////////////////// = End of TestDataset Class Definition = ///////////////////////////////// #

class TestDataLoader:
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.img_lr_path = cfg.DIR.IMAGE_LR_TEST_PATH
        # Load all files of the dataset
        self.samples = sorted(os.listdir(self.img_lr_path))

    def get_dataset(self, transforms=None):
        assert self.dataset_type == DatasetType.TEST
        files = []
        # Load data for each category
        for sample_idx, sample_name in enumerate(self.samples):
            # Get file path of img
            img_lr_path = os.path.join(self.img_lr_path,sample_name)
            if os.path.exists(img_lr_path):
                files.append({
                    'img_name': sample_name[:-4],
                    'img_lr'  : img_lr_path
                })

        print('[INFO] %s Complete collecting files for %s. Total test images: %d.' % (dt.now(), self.dataset_type.name, len(files)))
        return TestDataset(files, transforms)

# /////////////////////////////// = End of TestDataLoader Class Definition = /////////////////////////////// #


# Datasets MAP
DATASET_LOADER_MAPPING = {
    'DIV2K': SRDataLoader,
    'DIV2K_val': SRDataLoader,
    'Set5': SRDataLoader,
    'Set14': SRDataLoader,
    'BSD100': SRDataLoader,
    'Urban100': SRDataLoader,
    'Manga109': SRDataLoader,
    'Demo': TestDataLoader
}
