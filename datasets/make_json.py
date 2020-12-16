import json
import os
import io
import numpy as np


# For DIV2K, Set5, Set14, BSD100, Urban100, Manga109
file = io.open('div2k.json','w',encoding='utf-8')
samples = []

root = './DIV2K/DIV2K_train_HR_sub'
sample_list = sorted(os.listdir(root))
sample = [sample_list[i][:-4] for i in range(len(sample_list))]
sample_sub = []
for sam in sample:
    if not sam == ".DS_S":
        sample_sub.append(sam)
l = {'name': 'DIV2K', 'phase': 'train','sample': sample_sub}

samples.append(l)

js = json.dump(samples, file, sort_keys=True, indent=4)