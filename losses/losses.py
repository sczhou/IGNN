import torch
import torch.nn as nn
from config import cfg

def pixelLoss(output, target, type='L1'):
    if type == 'L1':
        loss = nn.L1Loss()
    if type == 'MSE':
        loss = nn.MSELoss()
    return loss(output, target)


def perceptualLoss(output, target, vggnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''
    weights = [1, 0.2, 0.04]
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm)
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss()

    loss = 0
    for i in range(len(features_real)):
        loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        loss = loss + loss_i * weights[i]

    return loss
