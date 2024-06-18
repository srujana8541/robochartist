""" Net Structure """

import logging
import numpy as np
from functools import partial
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
'''
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
'''


# Define Net Structure
class MLP1(nn.Module):
    def __init__(self, img_size=224, in_chans=3):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(img_size*img_size*in_chans, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # ([64, 1, 28, 28])->(64,784)
        x = x.view(x.size()[0], -1)
        print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class CNN(nn.Module):
    def __init__(self, img_H=96, img_W=180, in_chans=1):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_chans, out_channels=64,
                                             kernel_size=(5, 3), stride=(3, 1), padding=(5+img_H, 2), dilation=(3, 2)),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(), nn.MaxPool2d((2, 1)))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128,
                                             kernel_size=(5, 3), stride=(3, 1), padding=(5+img_H//2, 2), dilation=(3, 2)),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(), nn.MaxPool2d((2, 1)))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256,
                                             kernel_size=(5, 3), stride=(3, 1), padding=(5+img_H//4, 2), dilation=(3, 2)),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(), nn.MaxPool2d((2, 1)))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512,
                                             kernel_size=(5, 3), stride=(3, 1), padding=(5+img_H//8, 2), dilation=(3, 2)),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(), nn.MaxPool2d((2, 1)))

        self.fc1 = nn.Sequential(nn.Linear(in_features=184320 * 3, out_features=2),
                                 nn.Dropout(p=0.5), nn.Softmax(dim=1))

        self.weights_init_xavier()

    def weights_init_xavier(self):
        if isinstance(self, torch.nn.Linear) or isinstance(self, torch.nn.Conv2d):
            init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #    init.zeros_(self.bias)

    def forward(self, x):
        # mnist: ([batch_size, 1, 28, 28]) / cifar10: ([batch_size, 3, 32, 32])
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = x.view(x.size()[0], -1)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        return x


class CNN2(nn.Module):  # 3 channels, even for black-and-white pictures
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(184320, 2),
        )
        self.softmax = nn.Softmax(dim=1)
        self.weights_init_xavier()

    def weights_init_xavier(self):
        if isinstance(self, torch.nn.Linear) or isinstance(self, torch.nn.Conv2d):
            init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.reshape(-1, 3, 96, 180)
        # Automatically match batch size, channels=3, height=96, and width=180
        x = self.layer1(x)
        # print(x.shape)
        # Formulas:
        # For conv layer:
        # Output_height = ((Input_height + 2 * Padding_height - Dilation_height * (Kernel_height - 1) - 1) / Stride_height) + 1
        # Output_width = ((Input_width + 2 * Padding_width - Dilation_width * (Kernel_width - 1) - 1) / Stride_width) + 1
        # For maxpool layer:
        # Output_height = ((Input_height - Kernel_height) / Stride_height) + 1
        # Output_width = ((Input_width - Kernel_width) / Stride_width) + 1

        # x1 = x.detach().cpu().numpy()
        x = self.layer2(x)
        # x2 = x.detach().cpu().numpy()
        x = self.layer3(x)
        # x3 = x.detach().cpu().numpy()
        x = self.layer4(x)
        # x4 = x.detach().cpu().numpy()
        x = x.view(-1, 184320)
        x = self.fc1(x)
        x = self.softmax(x)

        return x

# TODO: can be deleted?
# class CNN2(nn.Module):  # 3 channels, even for black-and-white pictures
#     def __init__(self):
#         super(CNN2, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, )),
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             nn.Conv2d(256, 128, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             nn.MaxPool2d((2, 2), stride=(2, 2)),
#         )
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
#         )
#         self.fc1 = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(128*2*45, 16),
#             nn.Linear(16, 2),
#         )
#         self.softmax = nn.Softmax(dim=1)
#         self.weights_init_xavier()

#     def weights_init_xavier(self):
#         if isinstance(self, torch.nn.Linear) or isinstance(self, torch.nn.Conv2d):
#             init.xavier_uniform_(self.weight)

#     def forward(self, x):
#         x = x.reshape(-1, 3, 96, 180)
#         # Automatically match batch size, channels=3, height=96, and width=180
#         x = self.layer1(x)
#         # print(x.shape)
#         # Formula:
#         # For conv layer:
#         # Output_height = ((Input_height + 2 * Padding_height - Dilation_height * (Kernel_height - 1) - 1) / Stride_height) + 1
#         # Output_width = ((Input_width + 2 * Padding_width - Dilation_width * (Kernel_width - 1) - 1) / Stride_width) + 1
#         # For maxpool layer:
#         # Output_height = ((Input_height - Kernel_height) / Stride_height) + 1
#         # Output_width = ((Input_width - Kernel_width) / Stride_width) + 1

#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         # print(x.shape)
#         x = x.view(-1, 128*2*45)
#         x = self.fc1(x)
#         x = self.softmax(x)

#         return x

class CNN3(nn.Module):  # Use a larger convolution kernel: 30*3
    def __init__(self):
        super(CNN3, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(30, 3), stride=(3, 1), padding=(20, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(30, 3), stride=(3, 1), padding=(20, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(30, 3), stride=(3, 1), padding=(20, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(30, 3), stride=(3, 1), padding=(20, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(184320, 2),
        )
        self.softmax = nn.Softmax(dim=1)
        self.weights_init_xavier()

    def weights_init_xavier(self):
        if isinstance(self, torch.nn.Linear) or isinstance(self, torch.nn.Conv2d):
            init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.reshape(-1, 3, 96, 180)
        x = self.layer1(x)
        # print(x.shape)
        # For conv layer:
        # Output_height = ((Input_height + 2 * Padding_height - Dilation_height * (Kernel_height - 1) - 1) / Stride_height) + 1
        # Output_width = ((Input_width + 2 * Padding_width - Dilation_width * (Kernel_width - 1) - 1) / Stride_width) + 1
        # For maxpool layer:
        # Output_height = ((Input_height - Kernel_height) / Stride_height) + 1
        # Output_width = ((Input_width - Kernel_width) / Stride_width) + 1

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, 184320)
        x = self.fc1(x)
        x = self.softmax(x)

        return x

class CNN4(nn.Module):  # For images with higher definition 
    def __init__(self):
        super(CNN4, self).__init__()
        #  288, 540
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(15, 9), stride=(3, 2), padding=(20, 6)),
            # nn.Conv2d(32, 64, kernel_size=(3, 6), stride=(6, 3), padding=(20, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(15, 9), stride=(3, 2), padding=(20, 1)),
            nn.Conv2d(64, 128, kernel_size=(9, 6), stride=(3, 1), padding=(10, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.layer3 = nn.Sequential(
            # nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 1), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.85),
            nn.Linear(256*2*17, 2),  # 1320960
        )
        self.softmax = nn.Softmax(dim=1)
        self.weights_init_xavier()

    def weights_init_xavier(self):
        if isinstance(self, torch.nn.Linear) or isinstance(self, torch.nn.Conv2d):
            init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.reshape(-1, 3, 288, 540)
        # Automatically match batch size, channels=3, height=96, and width=180
        x = self.layer1(x)
        # print(x.shape)
        x = F.max_pool2d(x, (2, 1), stride=(2, 2))
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = F.max_pool2d(x, (2, 1), stride=(2, 2))
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = F.max_pool2d(x, (2, 1), stride=(2, 2))
        # print(x.shape)
        x = x.view(-1, 256*2*17)
        x = self.fc1(x)
        x = self.softmax(x)

        return x

class CNN5(nn.Module):
    def __init__(self):
        super(CNN5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(52480, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class ResNet(nn.Module):
    def __init__(self, img_H=180, img_W=96, in_chans=3):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.maxpool2 = nn.MaxPool2d(2, 2)
        # Since MaxPooling downsamples twice, the feature map is reduced to 1/4 * 1/4 of the original image
        self.fc1 = nn.Sequential(nn.Linear(in_features=64*int(img_H/4)*int(img_W/4), out_features=1000), nn.Dropout(p=0.4), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=1000, out_features=10), nn.Softmax(dim=1))
        self.weights_init_xavier()

    def weights_init_xavier(self):
        if isinstance(self, torch.nn.Linear) or isinstance(self, torch.nn.Conv2d):
            init.xavier_uniform_(self.weight)
        # if self.bias is not None:
            # init.zeros_(self.bias)

    def forward(self, x):
        # mnist: ([batch_size, 1, 28, 28]) / cifar10: ([batch_size, 3, 32, 32])
        # print(x.shape)
        # print(self.conv1(x).shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.maxpool1(x)
        # print(x.shape)
        # print(self.conv2(x).shape)
        x = self.conv2(x) + x
        x = self.maxpool2(x)
        # print(x.shape)
        # Merge two channels and keep the first dimension (batch_size)
        x = x.view(x.size()[0], -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        # print(x.shape)
        return x


""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""

'''
_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (my experiments)
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),

    # patch models, imagenet21k (weights ported from official Google JAX impl)
    'vit_base_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_huge_patch14_224_in21k': _cfg(
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # hybrid models (weights ported from official Google JAX impl)
    'vit_base_resnet50_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.9, first_conv='patch_embed.backbone.stem.conv'),
    'vit_base_resnet50_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0, first_conv='patch_embed.backbone.stem.conv'),

    # hybrid models (my experiments)
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),

    # deit models (FB weights)
    'vit_deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'vit_deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'vit_deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',),
    'vit_deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'),
    'vit_deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'),
    'vit_deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth', ),
    'vit_deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x




class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=4,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, 
                 pruning_loc=None, token_ratio=None, distill=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        print('## diff vit pruning method')
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


        self.distill = distill

        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        features = x[:, 1:]
        x = x[:, 0]
        x = self.pre_logits(x)
        x = self.head(x)
        if self.training:
            if self.distill:
                return x, features
            else:
                return x
        else:
            return x
'''
