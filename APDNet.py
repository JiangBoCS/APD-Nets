import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import kornia
from Regularize import Regularize_block
from modules import JointLayer

def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!' % act_type)
    return layer


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None

    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('[ERROR] Padding layer [%s] is not implemented!' % pad_type)
    return layer


def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0, \
               act_type='prelu', norm_type='bn', pad_type='zero', mode='CNA'):
    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        return sequential(p, conv, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        return sequential(act, p, conv)

def dconv_layer(in_channels, out_channels, kernel_size=6, stride=2, dilation=1, bias=True, padding=2, \
                act_type='prelu', norm_type='bn', pad_type='zero', mode='CNA'):
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        return sequential(p, conv, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        return sequential(act, p, conv)

class APDNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3,num_features=144, num_steps=2, num_groups=2,upscale_factor=2):
        super(APDNet, self).__init__()
        n_feat = num_features
        self.para1 = nn.Parameter(torch.randn(1, 16, 16))
        self.para2 = nn.Parameter(torch.randn(1, 32, 32))
        self.para3 = nn.Parameter(torch.randn(1, 64, 64))
        self.j1 = JointLayer(n_feat, 16)
        self.j2 = JointLayer(n_feat, 16)
        self.j3 = JointLayer(n_feat, 16)
        # self.para4 = nn.Parameter(torch.randn(1, 3, 3)).cuda()

        self.s1 = Regularize_block(n_feat, 128, 128, Gaussian_noise=True, Uniform_noise=True)
        self.s2 = Regularize_block(n_feat, 64, 64, Gaussian_noise=True, Uniform_noise=True)
        self.s3 = Regularize_block(n_feat, 32, 32, Gaussian_noise=True, Uniform_noise=True)
        self.s4 = Regularize_block(n_feat, 16, 16, Gaussian_noise=True, Uniform_noise=True)

        self.conv_1_0 = conv_layer(in_channels, n_feat, 3, 1, padding=1)
        self.conv_1_1 = conv_layer(n_feat, n_feat, 3, 1, padding=1)
        self.R = nn.LeakyReLU(0.05, inplace=True)
        self.conv_2_0 = conv_layer(n_feat, n_feat, 3, 1, padding=1)
        self.conv_2_1 = conv_layer(n_feat, n_feat, 6, 2, padding=2)
        self.conv_3_0 = conv_layer(n_feat, n_feat, 3, 1, padding=1)
        self.conv_3_1 = conv_layer(n_feat, n_feat, 6, 2, padding=2)
        self.conv_4_0 = conv_layer(n_feat, n_feat, 3, 1, padding=1)
        self.conv_4_1 = conv_layer(n_feat, n_feat, 6, 2, padding=2)
        self.conv_5_0 = conv_layer(n_feat, n_feat, 3, 1, padding=1)
        self.conv_5_1 = conv_layer(n_feat, n_feat, 6, 2, padding=2)

        self.dconv_1_0 = dconv_layer(n_feat, n_feat, 3, 1, padding=1)
        self.dconv_1_1 = dconv_layer(n_feat, n_feat, 6, 2, padding=2)
        self.dconv_2_0 = dconv_layer(n_feat, n_feat, 3, 1, padding=1)
        self.dconv_2_1 = dconv_layer(n_feat, n_feat, 6, 2, padding=2)
        self.dconv_3_0 = dconv_layer(n_feat, n_feat, 3, 1, padding=1)
        self.dconv_3_1 = dconv_layer(n_feat, n_feat, 6, 2, padding=2)
        self.dconv_4_0 = dconv_layer(n_feat, n_feat, 3, 1, padding=1)
        self.dconv_4_1 = dconv_layer(n_feat, n_feat, 6, 2, padding=2)
        self.out = nn.Conv2d(n_feat, out_channels, 1, 1)

    def forward(self, inputs):
        x = self.conv_1_1(self.R(self.conv_1_0(inputs)))
        f1 = self.s1(self.conv_2_1(self.conv_2_0(x)))
        f2 = self.s2(self.conv_3_1(self.conv_3_0(f1)))
        f3 = self.s3(self.conv_4_1(self.conv_4_0(f2)))
        f4 = self.s4(self.conv_5_1(self.conv_5_0(f3)))

        # d1 = self.R(self.dconv_1(f4 + torch.mul(self.para1, nn.AdaptiveAvgPool2d(1)(f4))))
        d1 = self.dconv_1_1(self.dconv_1_0(f4) + self.para1*self.j1(self.dconv_1_0(f4)) + self.para1*self.j1(self.dconv_1_0(f4))*self.j1(self.dconv_1_0(f4)))
        # d2 = self.R(self.dconv_2(d1 + f3 + torch.mul(self.para2, nn.AdaptiveAvgPool2d(1)(d1 + f3))))

        d2 = self.dconv_2_1(self.dconv_2_0(d1 + f3) + self.para2 * self.j2(self.dconv_2_0(d1 + f3)) + self.para2 * self.j2(self.dconv_2_0(d1 + f3)) * self.j2(self.dconv_2_0(d1 + f3)))

        # d3 = self.R(self.dconv_3(d2 + f2 + torch.mul(self.para3, nn.AdaptiveAvgPool2d(1)(d2 + f2))))
        d3 = self.dconv_3_1(self.dconv_3_0(d2 + f2) + self.para3 * self.j3(self.dconv_3_0(d2 + f2)) + self.para3 * self.j3(self.dconv_3_0(d2 + f2)) * self.j3(self.dconv_3_0(d2 + f2)))
        out = self.out(self.dconv_4_1(self.dconv_4_0(d3+f1)))

        return out


#if __name__ == '__main__':
#    M = APDNet(in_channels=3, out_channels=3,num_features=224, num_steps=2, num_groups=2,upscale_factor=2).cuda()
#    # y = M(x)
#    # print(y.shape)
#    from ptflops import get_model_complexity_info
#
#    flops, params = get_model_complexity_info(M, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
#    print('Flops:  ' + flops)
#    print('Params: ' + params)
    
    
    # C = 180, 12.02M
    # C = 224, 18.61M
    # C = 256, 24.29M


