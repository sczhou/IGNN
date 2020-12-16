#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import os
import cv2
import torch
import utils.network_utils as net_utils
from datetime import datetime as dt

from time import time

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def test_woGT(cfg, test_data_loader, net):
    # Testing loop
    n_batches = len(test_data_loader)

    # Batch average meterics
    test_time = net_utils.AverageMeter()
    cur_test_name = dt.now().isoformat()+'_'+cfg.DIR.DATASET_SCALE

    for batch_idx, (img_name, img_lr_s, img_lr) in enumerate(test_data_loader):
        # Switch models to testing mode
        net.eval()

        with torch.no_grad():
            # Get data from data loader
            img_lr_s, img_lr = [net_utils.var_or_cuda(img) for img in [img_lr_s, img_lr]]

            # Test runtime
            if cfg.TEST.RUNTIME: torch.cuda.synchronize()
            
            time_start = time()

            if cfg.TEST.ENSEMBLE:
                if cfg.TEST.CHOP:
                    forward_function = net_utils.Chop(model=net)
                else:
                    forward_function = net.forward
                img_out = net_utils.forward_x8(img_lr_s, img_lr, forward_function)

            elif cfg.TEST.CHOP:
                img_out = net_utils.Chop(model=net)(img_lr_s, img_lr)
            else:
                img_out = net(img_lr_s, img_lr)

            if cfg.TEST.RUNTIME: torch.cuda.synchronize()
            test_time.update(time() - time_start)

            img_out = net_utils.tensor2img(img_out[0], min_max = [0, cfg.DATA.RANGE])

            # Save image
            if cfg.TEST.SAVEIMG:
                img_save_dir = os.path.join(cfg.DIR.OUT_PATH, cfg.DATASET.DATASET_TEST_NAME, cur_test_name)
                if cfg.TEST.ENSEMBLE == True:
                    img_save_dir = img_save_dir+'_8'
                if not os.path.isdir(img_save_dir):
                    mkdir(img_save_dir)
                img_save_path = os.path.join(img_save_dir,img_name[0]+'.png')
                cv2.imwrite(img_save_path, img_out)
                print('[TEST] [{0}/{1}]\t BT {2}\t Saving: {3}'.format(batch_idx + 1, n_batches, test_time, img_save_path))
            else:
                print('[TEST] [{0}/{1}]\t BT {2}'.format(batch_idx + 1, n_batches, test_time))

