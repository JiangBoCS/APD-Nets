# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.autograd import Function


def Generate_noisy_weight(x, miu=0, sigma=1, alpha=None, Gaussian_noise=True, Uniform_noise=False):
    G_noise = 0
    U_noise = 0
    if Gaussian_noise and (not miu is None) and (not sigma is None):
        G_noise = (torch.cuda.FloatTensor(x.size(0), 1, 1, 1).normal_(0, 1)).to(miu.device)
        G_noise = G_noise*sigma+miu
    if Uniform_noise and (not alpha is None):
        U_noise = (torch.cuda.FloatTensor(x.size(0), 1, 1, 1).uniform_(-1, 1)).to(alpha.device)
        U_noise = U_noise*alpha
    return G_noise + U_noise


class Regularize_block(nn.Module):
    def __init__(self, channels, height, width, Gaussian_noise=True, Uniform_noise=False):
        super(Regularize_block, self).__init__()
        self.Gaussian_noise = Gaussian_noise
        self.Uniform_noise = Uniform_noise
        # self.drop_rate=drop_rate
        self.vector = nn.Parameter(nn.init.kaiming_normal_(torch.rand(1, 1, height, width)))
        self.channel_project = nn.Conv2d(1, channels, kernel_size=1, stride=1, padding=0, bias=False)
        if Gaussian_noise:
            self.norm_weight_miu = nn.Parameter(torch.cuda.FloatTensor([0]), requires_grad = True)
            self.norm_weight_sigma = nn.Parameter(torch.cuda.FloatTensor([1]), requires_grad = True)
        else:
            self.norm_weight_miu = None
            self.norm_weight_sigma = None
        if Uniform_noise:
            self.uniform_weight = nn.Parameter(torch.cuda.FloatTensor([1]), requires_grad = True)
        else:
            self.uniform_weight = None
        
    # def forward(self, x):
    #     expand_vector = self.channel_project(self.vector) #[1,c,h,w]
    #     if self.training:
    #         MCMC_weight = Generate_noisy_weight(x, self.norm_weight_miu, self.norm_weight_sigma, self.uniform_weight, self.Gaussian_noise, self.Uniform_noise) #[B,1,1,1]
    #     else:
    #         if (not self.norm_weight_miu is None):
    #             MCMC_weight = self.norm_weight_miu
    #         else:
    #             MCMC_weight = 0
    #     regularize_feature = MCMC_weight * expand_vector #[B,c,h,w]
    #     return x + regularize_feature

    def forward(self, x):
        expand_vector = self.channel_project(self.vector)   # [1,c,h,w]
        MCMC_weight = Generate_noisy_weight(x, self.norm_weight_miu, self.norm_weight_sigma, self.uniform_weight, self.Gaussian_noise, self.Uniform_noise) #[B,1,1,1]
#        print('self.norm_weight_miu',  self.norm_weight_miu)
#        print('self.norm_weight_sigma', self.norm_weight_sigma)
#        print('self.uniform_weight', self.uniform_weight)
        regularize_feature = torch.mul(expand_vector, MCMC_weight)   # [B,c,h,w]
        #print(regularize_feature.shape, x.shape)
        return x + regularize_feature

# x = Regularize_block(3, 64, 64, Gaussian_noise=True, Uniform_noise=False)
# print(x.shape)