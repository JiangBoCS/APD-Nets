
import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat
import time
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info

sys.path.append('/home/ma-user/work/uformer_for_denoise')

import scipy.io as sio
from utils.loader import get_test_data
import utils

from model import UNet,Uformer,Uformer_Cross,Uformer_CatCross

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--gt_dir', default='/cache/SIDD/val/', type=str, help='Directory of validation images')
parser.add_argument('--input_dir', default='/cache/SIDD/val/', type=str, help='Directory of validation images')
parser.add_argument('--save_in', default='/cache/SIDD/val/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/denoising/sidd/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./log/vit16_0701_1/models/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='vit', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', default=True, help='Save denoised images in result directory')#action='store_true', type=bool, 
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=4, help='number of data loading workers')
parser.add_argument('--token_embed', type=str,default='linear', help='linear/conv token embedding')
parser.add_argument('--token_mlp', type=str,default='fem', help='ffn/leff token mlp')
# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
parser.add_argument('--val_ps', type=int, default=128, help='patch size of training sample')
parser.add_argument('--noiseL', type=float, default =15,  help='image noisy level')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
args = parser.parse_args()

def padding(x, factor = 16):
    h, w = x.shape[2], x.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_ = F.pad(x, (0, padw, 0, padh), 'reflect')
    return input_, h, w

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)
img_options_val = {'patch_size':args.val_ps}
test_dataset = get_test_data(args.gt_dir, args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration= utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
#model_restoration.eval()
with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(test_loader, 0):
        rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))
        # inputs = data_test[1].cuda()
        # print(type(data_test[0]))
        shape = (data_test[0].cuda()).shape
        inputs = torch.cuda.FloatTensor(shape).normal_(mean=0, std=(args.noiseL)/255.) if torch.cuda.is_available() else torch.FloatTensor(shape).normal_(mean=0, std=(args.noiseL)/255.)
        inputs = inputs + data_test[0].cuda()

        rgb_noisy, h, w = padding(inputs, factor = 64)
        
        filenames = data_test[2]

        t0 = time.time()
        rgb_restored = model_restoration.forward(rgb_noisy)
        t1 = time.time()
        rgb_restored = rgb_restored[:, :, :h, :w]
        rgb_restored = torch.clamp(rgb_restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
        in_data = torch.clamp(inputs,0,1).cpu().numpy().squeeze().transpose((1,2,0))
        
        psnr = psnr_loss(rgb_restored, rgb_gt)
        ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True)
        
        print("[%d/%d] %s || PSNR(dB)/SSIM: %.2f/%.4f || Timer: %.4f sec ." % (ii+1, len(test_loader),
                                                                                       filenames[0],
                                                                                       psnr, ssim,
                                                                                       (t1 - t0)))
        
        psnr_val_rgb.append(psnr)
        ssim_val_rgb.append(ssim)


        if args.save_images:
            utils.save_img(os.path.join(args.result_dir,filenames[0]), img_as_ubyte(rgb_restored))
#            print(args.result_dir,filenames[0])
psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
print("PSNR: %f, SSIM: %f " %(psnr_val_rgb,ssim_val_rgb))
