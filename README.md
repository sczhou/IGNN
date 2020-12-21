# IGNN

Code repo for "Cross-Scale Internal Graph Neural Network for Image Super-Resolution" &nbsp; [[paper]](https://proceedings.neurips.cc/paper/2020/file/23ad3e314e2a2b43b4c720507cec0723-Paper.pdf) [[supp]](https://proceedings.neurips.cc/paper/2020/file/23ad3e314e2a2b43b4c720507cec0723-Supplemental.pdf)

<p align="center">
  <img width=95% src="https://user-images.githubusercontent.com/14334509/86379250-34450200-bcbd-11ea-9a85-aab4bc73cd2d.png">
</p>

## Prepare datasets
1 Download training dataset and test datasets from [here](https://drive.google.com/file/d/1fFBCXkUIgHkjqWiCeW7w-1TYHE0A2ZZF/view?usp=sharing).


2 Crop training dataset DIV2K to sub-images.
```
python ./datasets/prepare_DIV2K_subimages.py
```
Remember to modify the 'input_folder' and 'save_folder' in the above script.

## Dependencies and Installation
The denoising code is tested with Python 3.7, PyTorch 1.1.0 and Cuda 9.0 but is likely to run with newer versions of PyTorch and Cuda.

1 Create conda environment.
```
conda create --name ignn
conda activate ignn
conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch
```
2 Install PyInn.
```
pip install git+https://github.com/szagoruyko/pyinn.git@master
```
3 Install matmul_cuda
```
bash install.sh
```
4 Install other dependencies.
```
pip install -r requirements.txt
```

## Pretrained Models
Downloading the pretrained models from this [link](https://drive.google.com/drive/folders/1xS0jATn0MddZkLl2Rx9VPLh-U_rUxjt1?usp=sharing) and put them into ./ckpt

## Training
Use the following command to train the network:

```
python runner.py
        --gpu [gpu_id]\
        --phase 'train'\
        --scale [2/3/4]\
        --dataroot [dataset root]\
        --out [output path]
```
Use the following command to resume training the network:

```
python runner.py 
        --gpu [gpu_id]\
        --phase 'resume'\
        --weights './ckpt/IGNN_x[2/3/4].pth'\
        --scale [2/3/4]\
        --dataroot [dataset root]\
        --out [output path]
```
You can also use the following simple command with different settings in config.py:

```
python runner.py
```

## Testing
Use the following command to test the network on benchmark datasets (w/ GT):
```
python runner.py \
        --gpu [gpu_id]\
        --phase 'test'\
        --weights './ckpt/IGNN_x[2/3/4].pth'\
        --scale [2/3/4]\
        --dataroot [dataset root]\
        --testname [Set5, Set14, BSD100, Urban100, Manga109]\
        --out [output path]
```

Use the following command to test the network on your demo images (wo/ GT):
```
python runner.py \
        --gpu [gpu_id]\
        --phase 'test'\
        --weights './ckpt/IGNN_x[2/3/4].pth'\
        --scale [2/3/4]\
        --demopath [test folder path]\
        --out [output path]
```

You can also use the following simple command with different settings in config.py:

```
python runner.py
```

## Visual Results (x4)
For visual comparison on the 5 benchmarks, you can download our IGNN results from [here](https://drive.google.com/file/d/15x81tYQVpml4OvFqbA05mQQSRKL8phxz/view?usp=sharing).

### Some examples

![image](https://user-images.githubusercontent.com/14334509/86381317-c817cd80-bcbf-11ea-9b29-1f60ebfaa2e5.png)

![image](https://user-images.githubusercontent.com/14334509/86384957-129a4980-bcc2-11ea-9405-c81c3af6d01f.png)

## Citation

If you find our work useful for your research, please consider citing the following papers :)

```
@inproceedings{zhou2020cross,
title={Cross-scale internal graph neural network for image super-resolution},
author={Zhou, Shangchen and Zhang, Jiawei and Zuo, Wangmeng and Loy, Chen Change},
booktitle={Advances in Neural Information Processing Systems},
year={2020}
}
```
## Contact

We are glad to hear from you. If you have any questions, please feel free to contact shangchenzhou@gmail.com.

## License

This project is open sourced under MIT license.
