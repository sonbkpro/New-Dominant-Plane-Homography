# coding: utf-8
"""
Training script for Calibrated Dominant-Plane Consensus (CDPC) homography.

Key differences from Oneline-DLTv1/train.py:
  - Total loss = L_align + 1.0*L_triplet + 0.01*L_inv + 0.01*L_support
                         + 0.001*L_smooth + 0.05*L_calib
  - TensorBoard logs each loss component separately.
  - Model saved under train_log_CDPC/.
"""

import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime

from torch_homography_model import build_model
from dataset import TrainDataset
from utils import display_using_tensorboard
from losses import loss_temporal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_LOG_DIR = 'train_log_CDPC'
exp_name = os.path.abspath(os.path.join(os.path.dirname('__file__'), os.path.pardir))
exp_train_log_dir = os.path.join(exp_name, TRAIN_LOG_DIR)
LOG_DIR       = os.path.join(exp_train_log_dir, 'logs')
MODEL_SAVE_DIR = os.path.join(exp_train_log_dir, 'real_models')

writer = SummaryWriter(log_dir=LOG_DIR)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ('true', '1', 'yes', 'y')


def _maybe_join(base, path):
    return path if os.path.isabs(path) else os.path.join(base, path)


def _to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    train_path = os.path.join(exp_name, 'Data/Train_List.txt')
    net = build_model(args.model_name, pretrained=args.pretrained)

    if args.finetune:
        model_path = _maybe_join(exp_name, args.model_path)
        print('Loading checkpoint:', model_path)
        from collections import OrderedDict
        checkpoint = torch.load(model_path, map_location='cpu')
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint)
        else:
            state_dict = checkpoint
        cleaned = OrderedDict(
            (k[7:] if k.startswith('module.') else k, v)
            for k, v in state_dict.items()
        )
        model_dict = net.state_dict()
        matched = {k: v for k, v in cleaned.items() if k in model_dict}
        model_dict.update(matched)
        net.load_state_dict(model_dict)
        print('Loaded {}/{} tensors.'.format(len(matched), len(model_dict)))

    net = torch.nn.DataParallel(net)
    if torch.cuda.is_available():
        net = net.cuda()

    train_data = TrainDataset(
        data_path=train_path, exp_path=exp_name,
        patch_w=args.patch_size_w, patch_h=args.patch_size_h, rho=16,
        return_invalid=args.use_invalid,
        return_triplet=args.use_temporal,
    )
    train_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size,
        num_workers=args.cpus, shuffle=True, drop_last=True,
    )

    optimizer = optim.Adam(
        net.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    print('Start training — CDPC')
    PRINT_FREQ = 200
    SAVE_FREQ  = 4000
    glob_iter  = 0

    # Running sums for periodic logging
    _sums = {k: 0.0 for k in
             ('total', 'align', 'triplet', 'inv', 'support', 'smooth',
              'calib', 'rel_valid', 'rel_invalid', 'temp')}

    for epoch in range(args.max_epoch):
        net.train()
        print(epoch, 'lr={:.6f}'.format(scheduler.get_last_lr()[0]))

        for i, batch_value in enumerate(train_loader):

            # ---- Save model checkpoint ----
            if glob_iter % SAVE_FREQ == 0 and glob_iter != 0:
                fname = '{}_iter_{}.pth'.format(args.model_name, glob_iter)
                torch.save(net, os.path.join(MODEL_SAVE_DIR, fname))
                for name, layer in net.named_parameters():
                    if layer.requires_grad and layer.grad is not None:
                        writer.add_histogram(name + '_grad',
                                             layer.grad.cpu().data.numpy(), glob_iter)
                        writer.add_histogram(name + '_data',
                                             layer.cpu().data.numpy(), glob_iter)

            # ---- Batch preparation ----
            org_imges     = batch_value['org'].float()
            input_tesnors = batch_value['input'].float()
            patch_indices = batch_value['patch_indices'].float()
            h4p           = batch_value['h4p'].float()

            # Individual images for tensorboard display
            I          = org_imges[:, 0:1, ...]
            I2_ori_img = org_imges[:, 1:2, ...]
            I2_patch   = input_tesnors[:, 1:2, ...]

            org_imges     = _to_device(org_imges)
            input_tesnors = _to_device(input_tesnors)
            patch_indices = _to_device(patch_indices)
            h4p           = _to_device(h4p)
            I             = _to_device(I)
            I2_ori_img    = _to_device(I2_ori_img)
            I2_patch      = _to_device(I2_patch)

            # ---- Forward / backward ----
            optimizer.zero_grad()
            valid_label = torch.ones(org_imges.size(0), 1, device=org_imges.device)
            out = net(
                org_imges, input_tesnors, h4p, patch_indices,
                rel_label=valid_label,
                compute_geometric=True,
            )

            loss_total   = out['loss_total'].mean()
            loss_align_v = out['loss_align'].mean()
            loss_tri_v   = out['loss_triplet'].mean()
            loss_inv_v   = out['loss_inv'].mean()
            loss_sup_v   = out['loss_support'].mean()
            loss_smo_v   = out['loss_smooth'].mean()
            loss_cal_v   = out['loss_calib'].mean()
            loss_rel_valid_v = out['loss_rel'].mean()
            loss_rel_invalid_v = loss_total.new_tensor(0.0)
            loss_temp_v = loss_total.new_tensor(0.0)

            if args.use_invalid:
                invalid_org = _to_device(batch_value['invalid_org'].float())
                invalid_input = _to_device(batch_value['invalid_input'].float())
                invalid_patch_indices = _to_device(batch_value['invalid_patch_indices'].float())
                invalid_h4p = _to_device(batch_value['invalid_h4p'].float())
                invalid_label = torch.zeros(
                    invalid_org.size(0), 1, device=invalid_org.device,
                )
                out_invalid = net(
                    invalid_org, invalid_input, invalid_h4p, invalid_patch_indices,
                    rel_label=invalid_label,
                    compute_geometric=False,
                )
                loss_total = loss_total + out_invalid['loss_total'].mean()
                loss_rel_invalid_v = out_invalid['loss_rel'].mean()

            if args.use_temporal and 'triplet01_org' in batch_value:
                t01 = net(
                    _to_device(batch_value['triplet01_org'].float()),
                    _to_device(batch_value['triplet01_input'].float()),
                    _to_device(batch_value['triplet01_h4p'].float()),
                    _to_device(batch_value['triplet01_patch_indices'].float()),
                    compute_geometric=False,
                )
                t12 = net(
                    _to_device(batch_value['triplet12_org'].float()),
                    _to_device(batch_value['triplet12_input'].float()),
                    _to_device(batch_value['triplet12_h4p'].float()),
                    _to_device(batch_value['triplet12_patch_indices'].float()),
                    compute_geometric=False,
                )
                t02 = net(
                    _to_device(batch_value['triplet02_org'].float()),
                    _to_device(batch_value['triplet02_input'].float()),
                    _to_device(batch_value['triplet02_h4p'].float()),
                    _to_device(batch_value['triplet02_patch_indices'].float()),
                    compute_geometric=False,
                )
                triplet_available = _to_device(batch_value['triplet_available'].float()).mean()
                loss_temp_v = loss_temporal(t02['H_ab'], t12['H_ab'], t01['H_ab']) * triplet_available
                loss_total = loss_total + args.lambda_temp * loss_temp_v

            loss_total.backward()
            optimizer.step()

            # ---- Accumulate for printing ----
            _sums['total']   += loss_total.item()
            _sums['align']   += loss_align_v.item()
            _sums['triplet'] += loss_tri_v.item()
            _sums['inv']     += loss_inv_v.item()
            _sums['support'] += loss_sup_v.item()
            _sums['smooth']  += loss_smo_v.item()
            _sums['calib']   += loss_cal_v.item()
            _sums['rel_valid']   += loss_rel_valid_v.item()
            _sums['rel_invalid'] += loss_rel_invalid_v.item()
            _sums['temp']        += loss_temp_v.item()

            if i % PRINT_FREQ == 0 and i != 0:
                avgs = {k: v / PRINT_FREQ for k, v in _sums.items()}
                print(
                    'Ep[{:03d}/{:03d}] It[{:05d}/{:05d}] '
                    'Total={:.4f} Align={:.4f} Tri={:.4f} '
                    'Inv={:.4f} Sup={:.4f} Smo={:.4f} Cal={:.4f} '
                    'Rel+={:.4f} Rel-={:.4f} Temp={:.4f} '
                    'lr={:.2e}'.format(
                        epoch + 1, args.max_epoch, i + 1, len(train_loader),
                        avgs['total'], avgs['align'], avgs['triplet'],
                        avgs['inv'], avgs['support'], avgs['smooth'], avgs['calib'],
                        avgs['rel_valid'], avgs['rel_invalid'], avgs['temp'],
                        scheduler.get_last_lr()[0],
                    )
                )
                _sums = {k: 0.0 for k in _sums}

            # ---- TensorBoard images ----
            if glob_iter % 200 == 0:
                display_using_tensorboard(
                    I, I2_ori_img, I2_patch,
                    out['pred_I2_d'],
                    out['patch_2_res_d'],
                    out['pred_F2_d'],
                    out['mask_ap_d'],
                    out['residual_map_ab'][:1, ...],
                    writer,
                )

            # ---- TensorBoard scalars ----
            writer.add_scalars('Loss', {
                'total':   loss_total.item(),
                'align':   loss_align_v.item(),
                'triplet': loss_tri_v.item(),
                'inv':     loss_inv_v.item(),
                'support': loss_sup_v.item(),
                'smooth':  loss_smo_v.item(),
                'calib':   loss_cal_v.item(),
                'rel_valid': loss_rel_valid_v.item(),
                'rel_invalid': loss_rel_invalid_v.item(),
                'temp': loss_temp_v.item(),
            }, glob_iter)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], glob_iter)

            glob_iter += 1

        scheduler.step()

    print('Finished Training')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus',  type=int, default=2)
    parser.add_argument('--cpus',  type=int, default=8)

    parser.add_argument('--img_w',       type=int, default=640)
    parser.add_argument('--img_h',       type=int, default=360)
    parser.add_argument('--patch_size_h', type=int, default=315)
    parser.add_argument('--patch_size_w', type=int, default=560)

    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--max_epoch',   type=int,   default=30)
    parser.add_argument('--lr',          type=float, default=1e-4)

    parser.add_argument('--model_name',  type=str,  default='resnet34')
    parser.add_argument('--pretrained',  type=str2bool, default=False)
    parser.add_argument('--finetune',    type=str2bool, default=False)
    parser.add_argument('--model_path',  type=str,  default='')
    parser.add_argument('--use_invalid', type=str2bool, default=True,
                        help='Train reliability with generated invalid pairs.')
    parser.add_argument('--use_temporal', type=str2bool, default=False,
                        help='Enable optional temporal composition loss.')
    parser.add_argument('--lambda_temp', type=float, default=0.1)

    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    train(args)
