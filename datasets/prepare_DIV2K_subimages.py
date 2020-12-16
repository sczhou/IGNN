"""A multi-thread tool to crop large images to sub-images for faster IO."""
import os
import os.path as osp
import sys
import cv2
import time
from multiprocessing import Pool
import numpy as np
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from shutil import get_terminal_size
def main():
	mode = 'pair'  # single (one input folder) | pair (extract corresponding GT and LR pairs)
	opt = {}
	opt['n_thread'] = 20
	opt['compression_level'] = 3  # 3 is the default value in cv2
	# CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
	# compression time. If read raw images during training, use 0 for faster IO speed.
	if mode == 'single':
		opt['input_folder'] = './DIV2K/DIV2K_train_HR'
		opt['save_folder'] = './DIV2K/DIV2K_train_HR_sub'
		opt['crop_sz'] = 480  # the size of each sub-image
		opt['step'] = 240  # step of the sliding crop window
		opt['thres_sz'] = 48  # size threshold
		extract_single(opt)
	elif mode == 'pair':
		GT_folder = './DIV2K/DIV2K_train_HR'
		save_GT_folder = './DIV2K/DIV2K_train_HR_sub/'

		crop_sz = 480  # the size of each sub-image (GT)
		step = 240  # step of the sliding crop window (GT)
		thres_sz = 48  # size threshold

		print('process GT...')
		opt['input_folder'] = GT_folder
		opt['save_folder'] = save_GT_folder
		opt['crop_sz'] = crop_sz
		opt['step'] = step
		opt['thres_sz'] = thres_sz
		extract_single(opt)
		print('process LR...')
		scale_ratio_list = [2,3,4]
		
		for scale_ratio in scale_ratio_list:
			print('process LR-X'+str(scale_ratio)+'...')
			LR_folder = './DIV2K/DIV2K_train_LR_bicubic/x'+str(scale_ratio)
			save_LR_folder = './DIV2K/DIV2K_train_LR_bicubic_sub/x'+str(scale_ratio)
			###########################################################################################################
			# check that all the GT and LR images have correct scale ratio
			img_GT_list = _get_paths_from_images(GT_folder)
			img_LR_list = _get_paths_from_images(LR_folder)
			assert len(img_GT_list) == len(img_LR_list), 'different length of GT_folder and LR_folder.'
			for path_GT, path_LR in zip(img_GT_list, img_LR_list):
				img_GT = Image.open(path_GT)
				img_LR = Image.open(path_LR)
				w_GT, h_GT = img_GT.size
				w_LR, h_LR = img_LR.size
				assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR weight [{:d}] for {:s}.'.format( 
					w_GT, scale_ratio, w_LR, path_GT)
				assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR weight [{:d}] for {:s}.'.format( 
					w_GT, scale_ratio, w_LR, path_GT)

			# check crop size, step and threshold size
			assert crop_sz % scale_ratio == 0, 'crop size is not {:d}X multiplication.'.format(
				scale_ratio)
			assert step % scale_ratio == 0, 'step is not {:d}X multiplication.'.format(scale_ratio)
			assert thres_sz % scale_ratio == 0, 'thres_sz is not {:d}X multiplication.'.format(
				scale_ratio)

			opt['input_folder'] = LR_folder
			opt['save_folder'] = save_LR_folder
			opt['crop_sz'] = crop_sz // scale_ratio
			opt['step'] = step // scale_ratio
			opt['thres_sz'] = thres_sz // scale_ratio
			extract_single(opt)
			assert len(_get_paths_from_images(save_GT_folder)) == len(
				_get_paths_from_images(save_LR_folder)), 'different length of save_GT_folder and save_LR_folder.'
	else:
		raise ValueError('Wrong mode.')

#######################################################################################################################
def extract_single(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))

    img_list = _get_paths_from_images(input_folder)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, opt):
    crop_sz = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']
    img_name = osp.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            cv2.imwrite(
                osp.join(opt['save_folder'],
                         img_name.replace('.png', '_s{:03d}.png'.format(index))), crop_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    return 'Processing {:s} ...'.format(img_name)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            img_path = os.path.join(dirpath, fname)
            images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
