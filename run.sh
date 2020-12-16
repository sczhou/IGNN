#!/bin/bash
# srun --gres=gpu:2 --job-name=sr -w SG-IDC1-10-51-2-31 python -u runner.py

# srun --job-name=sr -w SG-IDC1-10-51-2-66 python -u runner.py --gpu 0,1
srun --job-name=sr -w SG-IDC1-10-51-2-66 python -u runner.py --gpu 0,1 --demodata '/mnt/lustre/sczhou/datasets/Set5/LR/x4'

# srun --job-name=sr -w SG-IDC1-10-51-2-66 python -u runner.py --gpu 2,3
# srun --job-name=sr -w SG-IDC1-10-51-2-66 python -u runner.py --gpu 4,5
# srun --job-name=sr -w SG-IDC1-10-51-2-66 python -u runner.py --gpu 6,7