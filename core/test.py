#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import os
import sys
import cv2
import torch
from collections import OrderedDict
import utils.network_utils as net_utils
from losses.metrics import *

from time import time

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def test(cfg, test_data_loader, net, Best_Epoch):
    # Testing loop
    n_batches = len(test_data_loader)

    # Batch average meterics
    test_time = net_utils.AverageMeter()
    PSNRs = net_utils.AverageMeter()
    SSIMs = net_utils.AverageMeter()

    test_results = OrderedDict()
    test_results['name'] = []
    test_results['ssim'] = []
    test_results['psnr'] = []

    for batch_idx, (img_name, img_lr_s, img_lr, img_hr) in enumerate(test_data_loader):
        # Switch models to testing mode
        net.eval()

        with torch.no_grad():
            # Get data from data loader
            img_lr_s, img_lr, img_hr = [net_utils.var_or_cuda(img) for img in [img_lr_s, img_lr, img_hr]]

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
            img_hr  = net_utils.tensor2img(img_hr[0], min_max = [0, cfg.DATA.RANGE])

            img_out_, img_hr_ = net_utils.crop_border([img_out, img_hr], cfg.CONST.SCALE)

            img_out_y = net_utils.bgr2ycbcr(img_out_ / 255., only_y=True)
            img_hr_y = net_utils.bgr2ycbcr(img_hr_ / 255., only_y=True)

            # Calculate PSNR
            PSNR = calculate_psnr(img_out_y * 255, img_hr_y * 255)
            PSNRs.update(PSNR, cfg.CONST.TEST_BATCH_SIZE)

            # Calculate SSIM
            SSIM = calculate_ssim(img_out_y * 255, img_hr_y * 255)
            SSIMs.update(SSIM.item(), cfg.CONST.TEST_BATCH_SIZE)

            test_results['name'].append(img_name[0])
            test_results['psnr'].append(PSNR)
            test_results['ssim'].append(SSIM)     
    
            # Save image
            if cfg.TEST.SAVEIMG:
                img_save_dir = os.path.join(cfg.DIR.OUT_PATH, cfg.DATASET.DATASET_TEST_NAME, cfg.DIR.DATASET_SCALE)
                if cfg.TEST.ENSEMBLE == True:
                    img_save_dir = img_save_dir+'_8'
                if not os.path.isdir(img_save_dir):
                    mkdir(img_save_dir)
                img_save_path = os.path.join(img_save_dir,img_name[0]+'.png')
                cv2.imwrite(img_save_path, img_out)
                print('[TEST] [{0}/{1}]\t BT {2}\t PSNR {3}\t SSIM {4}\t Saving: {5}'
                        .format(batch_idx + 1, n_batches, test_time, PSNRs, SSIMs, img_save_path))
            else:
                print('[TEST] [{0}/{1}]\t BT {2}\t PSNR {3}\t SSIM {4}'
                        .format(batch_idx + 1, n_batches, test_time, PSNRs, SSIMs))


    print('============================ RESULTS ===========================')
    print('[TEST] Average_PSNR: ' + str(PSNRs.avg))
    print('[TEST] [Best Epoch {0}] BatchTime_avg {1} PSNR_avg {2} SSIM_avg {3}\n'
          .format(Best_Epoch, test_time.avg, PSNRs.avg, SSIMs.avg))

    # Output test results to text file
    result_file = open(os.path.join(img_save_dir, 'test_result.txt'), 'w')
    sys.stdout = result_file
    print('[TEST] DATASET: ' + cfg.DATASET.DATASET_TEST_NAME)
    print('[TEST] Sample Number: ' + str(len(test_results['name'])))
    print('[TEST] Average PSNR: %.3f' % PSNRs.avg)
    print('[TEST] Average SSIM: %.3f' % SSIMs.avg)
    for i, name in enumerate(test_results['name']):
        print('[TEST]\t PSNR: %.3f \t SSIM: %.3f \t Name: %s' % 
            (test_results['psnr'][i], test_results['ssim'][i], name))
    result_file.close()

    return PSNRs.avg

