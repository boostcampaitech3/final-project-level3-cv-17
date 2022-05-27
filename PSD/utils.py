import os
import re
import time
import glob
import random
import wandb
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from math import log10
from math import exp

from models.GCA import GCANet
from models.FFA import FFANet
from models.MSBDN import MSBDNNet

from datasets.pretrain_datasets import TestData


def get_dark_channel(I, w):
    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    dc = maxpool(0 - I[:, :, :, :])

    return -dc

def get_atmosphere(I, dark_ch, p):
    B, _, H, W = dark_ch.shape
    num_pixel = int(p * H * W)
    flat_dc = dark_ch.resize(B, H * W)
    flat_I = I.resize(B, 3, H * W)
    index = torch.argsort(flat_dc, descending=True)[:, :num_pixel]
    A = torch.zeros((B, 3)).to('cuda')
    for i in range(B):
        A[i] = flat_I[i, :, index[i][torch.argsort(torch.max(flat_I[i][:, index[i]], 0)[0], descending=True)[0]]]

    return A[:, :, None, None]

def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return [ssim_map.mean()]
    else:
        return [ssim_map.mean(1).mean(1).mean(1)]
    
def ssim(img1, img2, window_size=11, size_average=True):
    img1=torch.clamp(img1,min=0,max=1)
    img2=torch.clamp(img2,min=0,max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def validation(net, net_name, val_data_loader, device, category, save_tag=False):

    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):
        if batch_id > 1:
            break
        with torch.no_grad():
            haze, haze_A, gt, image_name = val_data
            haze = haze.to(device)
            gt = gt.to(device)
            B, _, H, W = haze.shape

            if net_name == 'MSBDNNet':
                if haze.size()[2] % 16 != 0 or haze.size()[3] % 16 != 0:
                    haze = F.upsample(haze, [haze.size()[2] + 16 - haze.size()[2] % 16,
                                    haze.size()[3] + 16 - haze.size()[3] % 16], mode='bilinear')
                if gt.size()[2] % 16 != 0 or gt.size()[3] % 16 != 0:
                    gt = F.upsample(gt, [gt.size()[2] + 16 - gt.size()[2] % 16, 
                                    gt.size()[3] + 16 - gt.size()[3] % 16], mode='bilinear')
                out, out_J, out_T, out_A, out_I = net(haze, haze_A, True)
            else:
                out, out_J, out_T, out_A, out_I = net(haze, True)
            #T = net(haze)
            #dc = get_dark_channel(haze, 15)
            #A = get_atmosphere(haze, dc, 0.001).repeat_interleave(H*W).view(B, 3, H, W)
            #dehaze = ((haze - A) / T + A).clamp(0, 1)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(out_J, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(ssim(out_J, gt))

        # --- Save image --- #
        if save_tag:
            save_image(out_J, image_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(dehaze, image_name, category):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    for ind in range(batch_num):
        torchvision.utils.save_image(dehaze_images[ind], './{}_results/{}'.format(category, image_name[ind][:-3] + 'png'))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    with open('/output/{}_log.txt'.format(category), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)


def adjust_learning_rate(wandb, optimizer, epoch, category, lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 20 if category == 'indoor' else 10

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            return param_group['lr']
    else:
        for param_group in optimizer.param_groups:
            return param_group['lr']
            

def edge_compute(x):
    x_diffx = torch.abs(x[:,:,1:] - x[:,:,:-1])
    x_diffy = torch.abs(x[:,1:,:] - x[:,:-1,:])

    y = x.new(x.size())
    y.fill_(0)
    y[:,:,1:] += x_diffx
    y[:,:,:-1] += x_diffx
    y[:,1:,:] += x_diffy
    y[:,:-1,:] += x_diffy
    y = torch.sum(y,0,keepdim=True)/3
    y /= 4
    return y


def generate_test_images(net, TestData, num_epochs, chosen_epoch):
    epoch = 0
    net.eval()
    test_data_dir = '/data/nnice1216/unlabeled1/'
    test_data_loader = torch.utils.data.DataLoader(TestData(test_data_dir), batch_size=1, shuffle=False, num_workers=8)

    with torch.no_grad():
        for epoch in range(num_epochs):
            
            if epoch not in chosen_epoch:
                continue
                
            net.load_state_dict(torch.load('/code/dehazeproject/haze_current_temp_{}'.format(epoch)))
            
            output_dir = '/output/image_epoch{}/'.format(epoch)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for batch_id, val_data in enumerate(test_data_loader):
                if batch_id > 150:
                    break
                haze, haze_A, name = val_data
                print(batch_id, 'BEGIN!')

                B, _, H, W = haze.shape
                haze.to(device)
                haze_A.to(device)
                pred = net(haze, True)
                ts = torch.squeeze(pred.clamp(0, 1).cpu())

                vutils.save_image(ts, output_dir + name[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))
                print(name[0].split('.')[0] + 'DONE!')
                

def load_model(backbone, model_dir, device, device_ids):
    
    if backbone == 'GCANet':
        net = GCANet()
        net.to(device)
        net = nn.DataParallel(net, device_ids=device_ids)
        model_path = os.path.join(model_dir, 'PSD-GCANET')
        net.load_state_dict(torch.load(model_path))
        
    if backbone == 'FFANet':
        gps = 3
        blocks = 19
        net = FFANet(gps=gps,blocks=blocks)
        net.to(device)
        net = nn.DataParallel(net, device_ids=device_ids)
        model_path = os.path.join(model_dir, 'PSD-FFANET')
        net.load_state_dict(torch.load(model_path))
        
    if backbone == 'MSBDNNet':
        net = MSBDNNet()
        net.to(device)
        net = nn.DataParallel(net, device_ids=device_ids)
        model_path = os.path.join(model_dir, 'PSD-MSBDN')
        net.load_state_dict(torch.load(model_path))
        
    return net


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [
            re.search(rf"%s(\d+)" % path.stem, d) for d in dirs
        ]  
        i = [int(m.groups()[0]) for m in matches if m]  
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_unlabel_image_for_viz(unlabel_haze, unlabel_gt):
    train_unlabel_imgs = []
    train_unlabel_gts = []
    for train_unlabel_img, train_unlabel_gt in zip(unlabel_haze, unlabel_gt):
        train_unlabel_img = train_unlabel_img.permute(1,2,0).cpu().numpy()
        train_unlabel_gt = train_unlabel_gt.permute(1,2,0).cpu().numpy()
        train_unlabel_imgs.append(wandb.Image(train_unlabel_img))
        train_unlabel_gts.append(wandb.Image(train_unlabel_gt))
    
    return train_unlabel_imgs, train_unlabel_gts


def train_pred_image_for_viz(finetune_out, backbone_out):
    batch_b_out_imgs = []
    batch_f_out_imgs = []
    for f_out, b_out in zip(finetune_out, backbone_out):
        batch_b_out = b_out.detach().permute(1,2,0).cpu().numpy()
        batch_f_out = f_out.detach().permute(1,2,0).cpu().numpy()
        batch_b_out_imgs.append(wandb.Image(batch_b_out))
        batch_f_out_imgs.append(wandb.Image(batch_f_out))
    
    return batch_b_out_imgs, batch_f_out_imgs
