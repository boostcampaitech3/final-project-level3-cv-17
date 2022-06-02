import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.our_datasets import TrainData_label, ValData_label
from utils import to_psnr, increment_path, make_directory, load_model, pretrain_train_pred_image_for_viz, ssim, pretrain_val_pred_image_for_viz, set_seed
import math
from wandb_setup import wandb_login, wandb_init
from importlib import import_module
from tqdm import tqdm
import wandb
import warnings
from CR import *
warnings.filterwarnings("ignore")
#from perceptual import LossNetwork


def lr_schedule_cosdecay(t,T,init_lr=1e-4):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='DehazeFormer_m', help='Backbone model(GCANet/FFANet/MSBDNNet/DehazeFormer)')
    parser.add_argument('--category', type=str, default='outdoor', help='dataset type: indoor / outdoor') # outdoor only
    parser.add_argument('--seed', type=int, default=42, help='Enter random seed')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=1) # do not change
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--crop_size', type=int, default=256, help='size of random crop')
    parser.add_argument('--resize_size', type=int, default=1024, help='size of random crop')
    parser.add_argument('--all_T', type=int, default=100000)

    parser.add_argument('--work_dir', type=str, default='/opt/ml/input/final-project-level3-cv-17/PSD/work_dirs')
    parser.add_argument('--train_data_dir', type=str, default='/opt/ml/input/final-project-level3-cv-17/data/RESIDE-OTS')
    parser.add_argument('--val_data_dir', type=str, default='/opt/ml/input/final-project-level3-cv-17/data/RESIDE_SOTS_OUT')
    parser.add_argument('--pretrain_model_dir', type=str, default='/opt/ml/input/final-project-level3-cv-17/PSD/pretrained_model')

    parser.add_argument('--optimizer', type=str, default='AdamW', help='Enter optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Enter weight decay')
    parser.add_argument('--loss_vgg7_w', type=float, default=1, help='Enter weight decay')

    opt = parser.parse_known_args()[0]
    return opt

def train(opt, work_dir_exp, device, train_data_loader, val_data_loader, optimizer, net, criterion):
    best_PSNR, best_PSNR_epoch = 0.0, 0
    best_SSIM, best_SSIM_epoch = 0.0, 0
    for epoch in range(opt.num_epochs):
        torch.cuda.empty_cache()
        net.train()
        
        # Training ... 
        train_PSNR = []
        train_rec_loss1, train_rec_loss2, train_vgg_loss, train_total_loss = 0., 0., 0., 0.
        pbar = tqdm(train_data_loader, total=len(train_data_loader), desc=f"[Epoch {epoch+1}] Train")
        fin_bid = 0
        for batch_id, train_data in enumerate(pbar):
            if batch_id > 5000:
                break
            step_num = batch_id + epoch * 5000 + 1
            lr=lr_schedule_cosdecay(step_num, opt.all_T)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            haze, gt = train_data
            haze = haze.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()

            _, J, _, _, I = net(haze)
            #s, v = get_SV_from_HSV(J)
            #CAP_loss = F.smooth_l1_loss(s, v)

            if batch_id == 0:
                original_haze_img, reconstruct_haze_img, original_clear_img, pretrained_clear_img = pretrain_train_pred_image_for_viz(haze, I, gt, J)
                wandb.log({'original_haze_img':original_haze_img, 'reconstruct_haze_img':reconstruct_haze_img, 
                'original_clear_img':original_clear_img, 'pretrained_clear_img':pretrained_clear_img})

            Rec_Loss1 = F.smooth_l1_loss(J, gt)
            Rec_Loss2 = F.smooth_l1_loss(I, haze)
            loss_vgg7 = criterion(J, gt, haze)
            loss = Rec_Loss1 + Rec_Loss2 + loss_vgg7

            loss.backward()
            optimizer.step()

            train_PSNR.extend(to_psnr(J, gt))
            pbar.set_postfix(Epoch=f'{epoch}', gt_Rec_Loss=f" {Rec_Loss1:.3f}", gt_Rec_Loss2=f" {Rec_Loss2:.3f}", vgg_loss=f" {loss_vgg7:.3f}", Total_Loss=f" {loss:.3f}")
            
            if Rec_Loss1 != 0 : train_rec_loss1 += Rec_Loss1.item()
            if Rec_Loss2 != 0 : train_rec_loss2 += Rec_Loss2.item()
            if loss_vgg7 != 0 : train_vgg_loss += loss_vgg7.item()
            if loss != 0 : train_total_loss += loss.item()
            
            fin_bid+=1
            #if not (batch_id % 100):
            # print('Epoch: {}, Iteration: {}, Loss: {}, Rec_Loss1: {}, Rec_loss2: {}'.format(epoch, batch_id, loss, Rec_Loss1, Rec_Loss2))
        train_avg_psnr_per_epoch = sum(train_PSNR) / len(train_PSNR)
        wandb.log({
            'train/PSNR':train_avg_psnr_per_epoch,
            'train/RecLoss1':train_rec_loss1 / fin_bid,
            'train/RecLoss2':train_rec_loss2 / fin_bid,
            'train/VGGLoss':train_vgg_loss / fin_bid,
            'train/TotalLoss':train_total_loss / fin_bid,
            })
        
        if (epoch+1) % 5 == 0:
            torch.save(net.state_dict(), os.path.join(work_dir_exp, f'Epoch{epoch}.pth'))
    
        # Validation ...
        net.eval()

        val_PSNR, val_SSIM = [], []
        pbar = tqdm(val_data_loader, total=len(val_data_loader), desc=f"[Epoch {epoch+1}] Val")
        random_num = np.random.randint(0,len(val_data_loader))
        for b_id, val_data in enumerate(pbar):
            with torch.no_grad():
                haze, haze_A, gt, _ = val_data
                haze, gt = haze.to(device), gt.to(device)

                if opt.backbone in ['MSBDNNet','DehazeFormer_m']:
                    if haze.size()[2] % 16 != 0 or haze.size()[3] % 16 != 0:
                        haze = F.upsample(haze, [haze.size()[2] + 16 - haze.size()[2] % 16, haze.size()[3] + 16 - haze.size()[3] % 16], mode='bilinear')
                    if gt.size()[2] % 16 != 0 or gt.size()[3] % 16 != 0:
                        gt = F.upsample(gt, [gt.size()[2] + 16 - gt.size()[2] % 16, gt.size()[3] + 16 - gt.size()[3] % 16], mode='bilinear')
                    _, out_J, _, _, _ = net(haze, haze_A, True)
                else:
                    _, out_J, _, _, _ = net(haze, val=True)

                if b_id == random_num:
                    hazy_img, val_pred_img, gt_img = pretrain_val_pred_image_for_viz(haze, out_J, gt)
                    wandb.log({'hazy_img' : hazy_img, 'val_pred_img':val_pred_img, 'gt' : gt_img}, commit=False)

                val_PSNR.extend(to_psnr(out_J, gt))
                val_SSIM.extend(ssim(out_J, gt))
                avg_PSNR = sum(val_PSNR)/len(val_PSNR)
                avg_SSIM = sum(val_SSIM)/len(val_SSIM)

                
            pbar.set_postfix(
                Val_PSNR=f" {avg_PSNR:.3f}", Val_SSIM=f" {avg_SSIM:.3f}"
                )

        wandb.log({
            'val/PSNR':avg_PSNR, 'val/SSIM':avg_SSIM,
            })
        if best_PSNR < avg_PSNR:
            best_PSNR, best_PSNR_epoch = avg_PSNR, epoch
            best_PSNR_path = os.path.join(work_dir_exp, 'best_PSNR.pth')
            torch.save(net.state_dict(), best_PSNR_path)
        if best_SSIM < avg_SSIM:
            best_SSIM, best_SSIM_epoch = avg_SSIM, epoch
            best_SSIM_path = os.path.join(work_dir_exp, 'best_SSIM.pth')
            torch.save(net.state_dict(), best_SSIM_path)
        if epoch == opt.num_epochs:
            new_best_PSNR_path = best_PSNR_path[:-4] + f"_epoch{best_PSNR_epoch}.pth"
            new_best_SSIM_path = best_SSIM_path[:-4] + f"_epoch{best_SSIM_epoch}.pth"
            os.rename(best_PSNR_path, new_best_PSNR_path)
            os.rename(best_SSIM_path, new_best_SSIM_path)
            last_epoch_path = os.path.join(work_dir_exp, f'epoch{epoch}.pth')
            torch.save(net.state_dict(), last_epoch_path)


def main(opt):
    work_dir_exp = increment_path(os.path.join(opt.work_dir, 'exp'))
    make_directory(work_dir_exp)
    wandb_init(opt, work_dir_exp)
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = load_model(opt.backbone, '/opt/ml/input/final-project-level3-cv-17/PSD/pretrained_model', device, device_ids, type='pretrain')
    opt_module = getattr(import_module("torch.optim") ,opt.optimizer) #default : AdamW
    optimizer = opt_module(params=net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay) 
    train_data_loader = DataLoader(TrainData_label(opt.crop_size, opt.resize_size, opt.train_data_dir), batch_size=opt.train_batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_data_loader = DataLoader(ValData_label(opt.val_data_dir), batch_size=opt.val_batch_size, shuffle=False, pin_memory=True, num_workers=opt.num_workers, drop_last=True)
    print(">>> DATALOADER DONE!")
    torch.backends.cudnn.benchmark = True
    criterion = ContrastLoss()
    train(opt, work_dir_exp, device, train_data_loader, val_data_loader, optimizer, net, criterion)



if __name__ == '__main__':
    opt = get_parser()
    wandb_login()
    set_seed(opt.seed)
    main(opt)