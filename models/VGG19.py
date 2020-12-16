import torch
import torch.nn as nn
import torchvision.models
import warnings
from models.submodules import *

class VGG19(nn.Module):
    def __init__(self, feature_list=[2, 7, 14], requires_grad=True):
        super(VGG19, self).__init__()
        '''
        'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
        ]
         use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
        '''
        self.feature_list = feature_list
        vgg19 = torchvision.models.vgg19(pretrained=True)

        self.model = torch.nn.Sequential(*list(vgg19.features.children())[:self.feature_list[-1]+1])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)

        self.sub_mean = MeanShift(1.0, vgg_mean, vgg_std)

        for p in self.parameters():
            p.requires_grad = False


    def forward(self, x):
        """
        x : The input RGB tensor normalized to [0, 1].
        """
        x = x/255.
        if torch.any(x < 0.) or torch.any(x > 1.):
            warnings.warn('input tensor is not normalize to [0, 1].')

        x = self.sub_mean(x)
        features = []

        for i, layer in enumerate(list(self.model)):
            x = layer(x)
            if i in self.feature_list:
                features.append(x)
            if i == self.feature_list[-1]:
                if len(self.feature_list) == 1: return features[0]
                else: return features
        
