#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
import os
import utils.network_utils as net_utils
from losses.losses import *
from losses.metrics import *

from time import time

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def val(cfg, epoch_idx, val_data_loader, net, val_writer):
    Init_Epoch = epoch_idx

    n_batches = len(val_data_loader)

    # Batch average meterics
    batch_time = net_utils.AverageMeter()
    data_time = net_utils.AverageMeter()
    PSNRs = net_utils.AverageMeter()

    batch_end_time = time()

    for batch_idx, (_, img_lr_s, img_lr, img_hr) in enumerate(val_data_loader):
        data_time.update(time() - batch_end_time)
        # Switch models to validation mode
        net.eval()

        with torch.no_grad():
            # Get data from data loader
            img_lr_s, img_lr, img_hr = [net_utils.var_or_cuda(img) for img in [img_lr_s, img_lr, img_hr]]

            if cfg.TEST.CHOP:
                img_out = net_utils.Chop(model=net)(img_lr_s, img_lr)
            else:
                img_out = net(img_lr_s, img_lr)

            if batch_idx < cfg.VAL.VISUALIZATION_NUM:
                if epoch_idx == Init_Epoch:
                    img_lr_cpu = img_lr[0].cpu()/255.
                    img_hr_cpu = img_hr[0].cpu()/255.
                    val_writer.add_image(cfg.CONST.NAME+'/IMG_LR' + str(batch_idx + 1), img_lr_cpu, epoch_idx + 1)
                    val_writer.add_image(cfg.CONST.NAME+'/IMG_HR' + str(batch_idx + 1), img_hr_cpu, epoch_idx + 1)

                img_out_cpu = img_out[0].cpu().clamp(0,cfg.DATA.RANGE)/255.
                val_writer.add_image(cfg.CONST.NAME+'/IMG_OUT'+str(batch_idx+1), img_out_cpu, epoch_idx+1)

            img_out = net_utils.tensor2img(img_out[0], min_max = [0, cfg.DATA.RANGE])
            img_hr  = net_utils.tensor2img(img_hr[0], min_max = [0, cfg.DATA.RANGE])

            img_out, img_hr = net_utils.crop_border([img_out, img_hr], cfg.CONST.SCALE)

            img_out_y = net_utils.bgr2ycbcr(img_out / 255., only_y=True)
            img_hr_y = net_utils.bgr2ycbcr(img_hr / 255., only_y=True)

            PSNR = calculate_psnr(img_out_y * 255, img_hr_y * 255)
            PSNRs.update(PSNR, cfg.CONST.VAL_BATCH_SIZE)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            if (batch_idx + 1) % cfg.VAL.PRINT_FREQ == 0:
                print('[VAL] [Epoch {0}/{1}][Batch {2}/{3}]\t BT {4}\t DT {5}\t PSNR {6}'
                    .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time, data_time, PSNRs))

    # Add validation results to TensorBoard
    val_writer.add_scalar(cfg.CONST.NAME+'/PSNR_VAL', PSNRs.avg, epoch_idx + 1)

    print('============================ RESULTS ===========================')
    print('[VAL] Average_PSNR: ' + str(PSNRs.avg))
    print('[VAL] [Epoch {0}] BatchTime_avg {1} DataTime_avg {2} PSNR_avg {3}\n'
          .format(epoch_idx+1, batch_time.avg, data_time.avg, PSNRs.avg))
    return PSNRs.avg

