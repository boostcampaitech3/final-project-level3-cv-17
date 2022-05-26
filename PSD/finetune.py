import argparse
import copy
import wandb
import pyiqa
from tqdm import tqdm

from datasets.concat_dataset import ConcatDataset
from datasets.our_datasets import SynTrainData, RealTrainData_CLAHE, SynValData
from losses.energy_functions import energy_dc_loss
from losses.loss_functions import bright_channel, lwf_sky
from utils import *
from wandb_setup import wandb_login, wandb_init

import warnings
warnings.filterwarnings("ignore")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='MSBDNNet', help='Backbone model(GCANet/FFANet/MSBDNNet)')
    parser.add_argument('--category', type=str, default='outdoor', help='dataset type: indoor / outdoor') # outdoor only

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=1) # do not change
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--crop_size', type=int, default=256, help='size of random crop')

    parser.add_argument('--work_dir', type=str, default='/opt/ml/final-project-level3-cv-17/PSD/work_dirs')
    parser.add_argument('--label_dir', type=str, default='/opt/ml/final-project-level3-cv-17/data/MRFID')
    parser.add_argument('--unlabel_dir', type=str, default='/opt/ml/final-project-level3-cv-17/data/RESIDE_RTTS')
    parser.add_argument('--val_dir', type=str, default='/opt/ml/final-project-level3-cv-17/data/RESIDE_SOTS_OUT')
    parser.add_argument('--pretrain_model_dir', type=str, default='/opt/ml/final-project-level3-cv-17/PSD/pretrained_model')

    parser.add_argument('--lambda_dc', type=float, default=1e-3)
    parser.add_argument('--lambda_bc', type=float, default=0.05)
    parser.add_argument('--lambda_CLAHE', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=1)
    parser.add_argument('--lambda_lwf_label', type=float, default=1)
    parser.add_argument('--lambda_lwf_unlabel', type=float, default=1)
    parser.add_argument('--lambda_lwf_sky', type=float, default=1)

    opt = parser.parse_known_args()[0]
    return opt


def train(opt, work_dir_exp, device, train_data_loader, val_data_loader, net, net_o, loss_dc, optimizer,
          metric_PSNR, metric_SSIM, metric_NIQE, metric_BRISQUE, metric_NIMA):
    best_PSNR, best_PSNR_epoch = 0.0, 0
    best_SSIM, best_SSIM_epoch = 0.0, 0

    for epoch in range(opt.num_epochs):
        # --- train --- #
        train_loss, train_DCP_loss, train_BCP_loss, train_CLAHE_loss, train_Rec_loss, train_LwF_loss = 0, 0, 0, 0, 0, 0
        learning_rate = adjust_learning_rate(wandb, optimizer, epoch, category=opt.category, lr_decay=0.5) # outdoor: decay_step=10
        wandb.log({'learning_rate':learning_rate}, commit=False)
        
        pbar = tqdm(train_data_loader, total=len(train_data_loader), desc=f"[Epoch {epoch+1}] Train")
        start_time = time.time()
        for b_id, (label_train_data, unlabel_train_data) in enumerate(pbar):
            
            label_haze, label_gt = label_train_data
            unlabel_haze, unlabel_gt = unlabel_train_data
            label_haze, label_gt = label_haze.to(device), label_gt.to(device)
            unlabel_haze, unlabel_gt = unlabel_haze.to(device), unlabel_gt.to(device)
            # print(f'>>> label_haze shape : {label_haze.shape}, label_gt : {label_gt.shape}, unlabel_haze.shape : {unlabel_haze.shape}, unlabel_gt.shape : {unlabel_gt.shape}')
            # --- viz unlabel data --- #
            if b_id == 0:
                train_unlabel_imgs, train_unlabel_clahe = train_unlabel_image_for_viz(unlabel_haze, unlabel_gt)
                wandb.log({'train_unlabel_hazy_image':train_unlabel_imgs, 'train_unlabel_dehaze_with_clahe_image':train_unlabel_clahe}, commit=False)

            optimizer.zero_grad()
            net.train()
            
            out_label, J_label, T_label, _, _ = net(label_haze)
            out_label_o, J_label_o, T_label_o, _, _ = net_o(label_haze)
            out, J, T, A, I = net(unlabel_haze)
            out_o, J_o, _, _, _ = net_o(unlabel_haze)
            I2 = T * unlabel_gt + (1 - T) * A

            finetune_out = torch.squeeze(J.clamp(0, 1).cpu())
            backbone_out = torch.squeeze(J_o.clamp(0, 1).cpu())
            # I2 shape : torch.Size([6, 3, 256, 256]), out shape : torch.Size([6, 64, 256, 256]), out_o.shape : torch.Size([6, 64, 256, 256])
            # print(f'>>> I2 shape : {I2.shape}, finetune_out : {finetune_out.shape}, backbone_out.shape : {backbone_out.shape}')
            # --- viz output --- #
            if b_id == 0:
                batch_b_out_imgs, batch_f_out_imgs = val_pred_image_for_viz(finetune_out, backbone_out)
                wandb.log({'backbone_output':batch_b_out_imgs, 'finetune_output':batch_f_out_imgs}, commit=False)

            # --- losses --- #
            energy_dc_loss = loss_dc(unlabel_haze, T)
            bc_loss = bright_channel(unlabel_haze, T)
            CLAHE_loss = F.smooth_l1_loss(I2, unlabel_haze)
            rec_loss = F.smooth_l1_loss(I, unlabel_haze)
            lwf_loss_sky = lwf_sky(unlabel_haze, J, J_o)
            lwf_loss_label = F.smooth_l1_loss(out_label, out_label_o)
            lwf_loss_unlabel = F.smooth_l1_loss(out, out_o)
            
            total_loss = opt.lambda_dc*energy_dc_loss + opt.lambda_bc*bc_loss + opt.lambda_CLAHE*CLAHE_loss + opt.lambda_rec*rec_loss
            total_loss += opt.lambda_lwf_sky*lwf_loss_sky + opt.lambda_lwf_label*lwf_loss_label + opt.lambda_lwf_unlabel*lwf_loss_unlabel
            total_loss.backward()
            optimizer.step()

            if total_loss != 0 : train_loss += total_loss.item()
            if energy_dc_loss != 0 : train_DCP_loss += opt.lambda_dc*energy_dc_loss.item()
            if bc_loss != 0 : train_BCP_loss += opt.lambda_bc*bc_loss.item()
            if CLAHE_loss != 0 : train_CLAHE_loss += opt.lambda_CLAHE*CLAHE_loss.item()
            if rec_loss != 0 : train_Rec_loss += opt.lambda_rec*rec_loss.item()
            if lwf_loss_sky != 0: train_LwF_loss += opt.lambda_lwf_sky*lwf_loss_sky.item()
            if lwf_loss_label != 0: train_LwF_loss += opt.lambda_lwf_sky*lwf_loss_label.item()
            if lwf_loss_unlabel != 0: train_LwF_loss += opt.lambda_lwf_sky*lwf_loss_unlabel.item()

            pbar.set_postfix(
                Total_Loss=f" {train_loss/(b_id+1):.3f}", DCP_Loss=f" {train_DCP_loss/(b_id+1):.3f}", BCP_Loss=f" {train_BCP_loss/(b_id+1):.3f}",
                CLAHE_Loss=f" {train_CLAHE_loss/(b_id+1):.3f}", Rec_Loss=f" {train_Rec_loss/(b_id+1):.3f}", LwF_Loss=f" {train_LwF_loss/(b_id+1):.3f}",
                )
        
        run_time = time.time() - start_time
        wandb.log({'train/total_loss':train_loss/(b_id+1), 'train/DCP_loss':train_DCP_loss/(b_id+1), 'train/BCP_loss':train_BCP_loss/(b_id+1),
                   'train/CLAHE_loss':train_CLAHE_loss/(b_id+1), 'train/Rec_loss':train_Rec_loss/(b_id+1), 'train/LwF_loss':train_LwF_loss/(b_id+1),
                   'train_run_time':run_time,
                   }, commit=False)


        # --- evaluation --- #
        net.eval()

        val_PSNR, val_SSIM = [], []
        val_PSNR_score, val_SSIM_score, val_NIQE_score, val_BRISQUE_score, val_NIMA_score = 0, 0, 0, 0, 0
        start_time = time.time()
        pbar = tqdm(val_data_loader, total=len(val_data_loader), desc=f"[Epoch {epoch+1}] Val")
        for b_id, val_data in enumerate(pbar):

            with torch.no_grad():
                haze, haze_A, gt, image_name = val_data
                haze, gt = haze.to(device), gt.to(device)
                B, _, H, W = haze.shape

                if opt.backbone == 'MSBDNNet':
                    if haze.size()[2] % 16 != 0 or haze.size()[3] % 16 != 0:
                        haze = F.upsample(haze, [haze.size()[2] + 16 - haze.size()[2] % 16, haze.size()[3] + 16 - haze.size()[3] % 16], mode='bilinear')
                    if gt.size()[2] % 16 != 0 or gt.size()[3] % 16 != 0:
                        gt = F.upsample(gt, [gt.size()[2] + 16 - gt.size()[2] % 16, gt.size()[3] + 16 - gt.size()[3] % 16], mode='bilinear')
                    out, out_J, out_T, out_A, out_I = net(haze, haze_A, True)
                else:
                    out, out_J, out_T, out_A, out_I = net(haze, True)
            
                val_PSNR.extend( to_psnr(out_J, gt) )
                val_SSIM.extend( ssim(out_J, gt) )
                # val_PSNR_score += metric_PSNR(out_J, gt).item()
                # val_SSIM_score += metric_SSIM(out_J, gt).item()
                val_NIQE_score += metric_NIQE(out_J).item()
                val_BRISQUE_score += metric_BRISQUE(out_J).item()
                val_NIMA_score += metric_NIMA(out_J).item()

                avg_PSNR = sum(val_PSNR)/len(val_PSNR)
                avg_SSIM = sum(val_SSIM)/len(val_SSIM)
                
            pbar.set_postfix(
                Val_PSNR=f" {avg_PSNR:.3f}", Val_SSIM=f" {avg_SSIM:.3f}",
                # Val_PSNR_pyiqa=f" {val_PSNR_score/(b_id+1):.3f}", Val_SSIM_pyiqa=f" {val_SSIM_score/(b_id+1):.3f}",
                Val_NIQE=f" {val_NIQE_score/(b_id+1):.3f}", Val_BRISQUE=f" {val_BRISQUE_score/(b_id+1):.3f}", Val_NIMA=f" {val_NIMA_score/(b_id+1):.3f}",
                )

            # --- Save image --- #
            # save_image(out_J, image_name, opt.category)

        run_time = time.time() - start_time
        wandb.log({
            'val/PSNR':avg_PSNR, 'val/SSIM':avg_SSIM, 'val_run_time':run_time,
            # 'val/PSNR_pyiqa':val_PSNR_score/(b_id+1), 'val/SSIM_pyiqa':val_SSIM_score/(b_id+1),
            'val/NIQE':val_NIQE_score/(b_id+1), 'val/BRISQUE':val_BRISQUE_score/(b_id+1), 'val/NIMA':val_NIMA_score/(b_id+1),
            })

        # --- Save model --- #
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

    # --- output test images --- #
    # generate_test_images(net, TestData, opt.num_epochs, (0, opt.num_epochs - 1))


def main(opt):
    # --- setup --- #
    work_dir_exp = increment_path(os.path.join(opt.work_dir, 'exp'))
    make_directory(work_dir_exp)
    wandb_init(opt, work_dir_exp)

    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- dataloader --- #
    crop_size = [opt.crop_size, opt.crop_size]

    train_data_loader = torch.utils.data.DataLoader(
                    ConcatDataset(
                        SynTrainData(crop_size, opt.label_dir),
                        RealTrainData_CLAHE(crop_size, opt.unlabel_dir)
                    ),
                    batch_size=opt.train_batch_size,
                    shuffle=False,
                    num_workers=opt.num_workers,
                    pin_memory=True,
                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(
                    SynValData(opt.val_dir),
                    batch_size=opt.val_batch_size,
                    shuffle=False,
                    num_workers=opt.num_workers,
                    pin_memory=True,
                    )

    # --- model --- #
    net = load_model(opt.backbone, opt.pretrain_model_dir, device, device_ids)
    net_o = copy.deepcopy(net)
    net_o.eval()

    loss_dc = energy_dc_loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    metric_PSNR = pyiqa.create_metric('psnr').to(device)
    metric_SSIM = pyiqa.create_metric('ssim').to(device)
    metric_NIQE = pyiqa.create_metric('niqe').to(device)
    metric_BRISQUE = pyiqa.create_metric('brisque').to(device)
    metric_NIMA = pyiqa.create_metric('nima').to(device)

    train(opt, work_dir_exp, device, train_data_loader, val_data_loader, net, net_o, loss_dc, optimizer,
          metric_PSNR, metric_SSIM, metric_NIQE, metric_BRISQUE, metric_NIMA)


if __name__ == '__main__':
    wandb_login()
    set_seed(42)
    opt = get_parser()
    main(opt)