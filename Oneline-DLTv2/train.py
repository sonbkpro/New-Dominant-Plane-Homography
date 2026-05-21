# coding: utf-8
"""
Training script for Calibrated Dominant-Plane Consensus (CDPC) homography.

Key differences from Oneline-DLTv1/train.py:
  - Strict Oneline CDPC: learns H_ab only; loss_inv is logged as zero.
  - Professional adaptation schedule: train first with a v1-style triplet
    objective, then ramp the heteroscedastic alignment NLL after the feature
    distances have a stable scale.
  - Weight decay is disabled on BN affine params, biases and the geometry
    head (fc) so the optimiser can't shrink them toward zero.
  - TensorBoard logs raw and weighted loss components separately.
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


def _scheduled_weight(base_weight, start_iter, warmup_iters, global_iter):
    if global_iter < start_iter:
        return 0.0
    if warmup_iters <= 0:
        return base_weight
    progress = float(global_iter - start_iter + 1) / float(warmup_iters)
    return base_weight * min(1.0, progress)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    train_path = os.path.join(exp_name, 'Data/Train_List.txt')
    net = build_model(
        args.model_name,
        pretrained=args.pretrained,
        lambda_align=args.lambda_align,
        triplet_margin=args.triplet_margin,
        lambda_triplet=args.lambda_triplet,
        lambda_support=args.lambda_support,
        lambda_smooth=args.lambda_smooth,
        lambda_offset=args.lambda_offset,
        lambda_rel=args.lambda_rel,
        lambda_temp=args.lambda_temp,
        fc_init_std=args.fc_init_std,
    )

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

    # Parameter groups: shrink only multi-dim conv/linear weights.  BN affine
    # params, all biases, and the geometry head (fc) are weight-decay-free —
    # weight decay on those parameters is the standard accelerant for
    # feature-magnitude collapse and zero-pinning of the offset head.
    decay_params, no_decay_params = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith('fc.weight') or name.endswith('fc.bias'):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = optim.Adam(
        [{'params': decay_params,    'weight_decay': 1e-4},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=args.lr, amsgrad=True,
    )
    print('Optimizer: {} params with WD=1e-4, {} params with WD=0.'.format(
        len(decay_params), len(no_decay_params)))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    print('Start training — CDPC')
    PRINT_FREQ = 200
    SAVE_FREQ  = 4000
    glob_iter  = 0

    # Running sums for periodic logging
    _sums = {k: 0.0 for k in
             ('total', 'align', 'align_weighted',
              'triplet', 'triplet_weighted',
              'inv', 'support', 'smooth',
              'offset', 'rel_valid', 'rel_invalid', 'temp',
              'align_weight', 'triplet_weight',
              'triplet_d_pos', 'triplet_d_neg',
              'triplet_gap', 'triplet_margin_gap', 'triplet_active',
              'q_mean', 'q_std',
              'sigma_mean', 'log_sigma_mean',
              'log_sigma_min', 'log_sigma_max')}

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
            current_align_weight = _scheduled_weight(
                args.lambda_align,
                args.align_start_iters,
                args.align_warmup_iters,
                glob_iter,
            )
            current_triplet_weight = _scheduled_weight(
                args.lambda_triplet,
                args.triplet_start_iters,
                args.triplet_warmup_iters,
                glob_iter,
            )
            out = net(
                org_imges, input_tesnors, h4p, patch_indices,
                rel_label=valid_label,
                compute_geometric=True,
                align_weight=current_align_weight,
                triplet_weight=current_triplet_weight,
            )

            loss_total   = out['loss_total'].mean()
            loss_align_v = out['loss_align'].mean()
            loss_align_weighted_v = out['loss_align_weighted'].mean()
            loss_tri_v   = out['loss_triplet'].mean()
            loss_tri_weighted_v = out['loss_triplet_weighted'].mean()
            loss_inv_v   = out['loss_inv'].mean()
            loss_sup_v   = out['loss_support'].mean()
            loss_smo_v   = out['loss_smooth'].mean()
            loss_off_v   = out['loss_offset'].mean()
            loss_rel_valid_v = out['loss_rel'].mean()
            loss_rel_invalid_v = loss_total.new_tensor(0.0)
            loss_temp_v = loss_total.new_tensor(0.0)
            align_weight_v = out['align_weight'].mean()
            triplet_weight_v = out['triplet_weight'].mean()
            triplet_d_pos_v = out['triplet_d_pos'].mean()
            triplet_d_neg_v = out['triplet_d_neg'].mean()
            triplet_gap_v = out['triplet_gap'].mean()
            triplet_margin_gap_v = out['triplet_margin_gap'].mean()
            triplet_active_v = out['triplet_active'].mean()
            q_mean_v = out['q_mean'].mean()
            q_std_v = out['q_std'].mean()
            sigma_mean_v = out['sigma_mean'].mean()
            log_sigma_mean_v = out['log_sigma_mean'].mean()
            log_sigma_min_v = out['log_sigma_min'].mean()
            log_sigma_max_v = out['log_sigma_max'].mean()

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
                    align_weight=current_align_weight,
                    triplet_weight=current_triplet_weight,
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
                    align_weight=current_align_weight,
                    triplet_weight=current_triplet_weight,
                )
                t12 = net(
                    _to_device(batch_value['triplet12_org'].float()),
                    _to_device(batch_value['triplet12_input'].float()),
                    _to_device(batch_value['triplet12_h4p'].float()),
                    _to_device(batch_value['triplet12_patch_indices'].float()),
                    compute_geometric=False,
                    align_weight=current_align_weight,
                    triplet_weight=current_triplet_weight,
                )
                t02 = net(
                    _to_device(batch_value['triplet02_org'].float()),
                    _to_device(batch_value['triplet02_input'].float()),
                    _to_device(batch_value['triplet02_h4p'].float()),
                    _to_device(batch_value['triplet02_patch_indices'].float()),
                    compute_geometric=False,
                    align_weight=current_align_weight,
                    triplet_weight=current_triplet_weight,
                )
                triplet_available = _to_device(batch_value['triplet_available'].float()).mean()
                loss_temp_v = loss_temporal(t02['H_ab'], t12['H_ab'], t01['H_ab']) * triplet_available
                loss_total = loss_total + args.lambda_temp * loss_temp_v

            loss_total.backward()
            optimizer.step()

            # ---- Accumulate for printing ----
            _sums['total']   += loss_total.item()
            _sums['align']   += loss_align_v.item()
            _sums['align_weighted'] += loss_align_weighted_v.item()
            _sums['triplet'] += loss_tri_v.item()
            _sums['triplet_weighted'] += loss_tri_weighted_v.item()
            _sums['inv']     += loss_inv_v.item()
            _sums['support'] += loss_sup_v.item()
            _sums['smooth']  += loss_smo_v.item()
            _sums['offset']  += loss_off_v.item()
            _sums['rel_valid']   += loss_rel_valid_v.item()
            _sums['rel_invalid'] += loss_rel_invalid_v.item()
            _sums['temp']        += loss_temp_v.item()
            _sums['align_weight'] += align_weight_v.item()
            _sums['triplet_weight'] += triplet_weight_v.item()
            _sums['triplet_d_pos'] += triplet_d_pos_v.item()
            _sums['triplet_d_neg'] += triplet_d_neg_v.item()
            _sums['triplet_gap'] += triplet_gap_v.item()
            _sums['triplet_margin_gap'] += triplet_margin_gap_v.item()
            _sums['triplet_active'] += triplet_active_v.item()
            _sums['q_mean'] += q_mean_v.item()
            _sums['q_std'] += q_std_v.item()
            _sums['sigma_mean'] += sigma_mean_v.item()
            _sums['log_sigma_mean'] += log_sigma_mean_v.item()
            _sums['log_sigma_min'] += log_sigma_min_v.item()
            _sums['log_sigma_max'] += log_sigma_max_v.item()

            if i % PRINT_FREQ == 0 and i != 0:
                avgs = {k: v / PRINT_FREQ for k, v in _sums.items()}
                print(
                    'Ep[{:03d}/{:03d}] It[{:05d}/{:05d}] '
                    'Total={:.4f} Align={:.4f} A*={:.4f} Tri={:.4f} T*={:.4f} '
                    'Inv={:.4f} Sup={:.4f} Smo={:.4f} Off={:.4f} '
                    'Rel+={:.4f} Rel-={:.4f} Temp={:.4f} '
                    'Aw={:.3f} TriW={:.3f} Dp={:.4f} Dn={:.4f} Gap={:.4f} '
                    'HAct={:.3f} Qm={:.3f} Qs={:.3f} Sig={:.4f} '
                    'LogS={:.2f}[{:.2f},{:.2f}] '
                    'lr={:.2e}'.format(
                        epoch + 1, args.max_epoch, i + 1, len(train_loader),
                        avgs['total'], avgs['align'], avgs['align_weighted'],
                        avgs['triplet'], avgs['triplet_weighted'],
                        avgs['inv'], avgs['support'], avgs['smooth'],
                        avgs['offset'], avgs['rel_valid'], avgs['rel_invalid'], avgs['temp'],
                        avgs['align_weight'], avgs['triplet_weight'],
                        avgs['triplet_d_pos'], avgs['triplet_d_neg'],
                        avgs['triplet_gap'], avgs['triplet_active'],
                        avgs['q_mean'], avgs['q_std'],
                        avgs['sigma_mean'], avgs['log_sigma_mean'],
                        avgs['log_sigma_min'], avgs['log_sigma_max'],
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
                'align_weighted': loss_align_weighted_v.item(),
                'triplet': loss_tri_v.item(),
                'triplet_weighted': loss_tri_weighted_v.item(),
                'inv':     loss_inv_v.item(),
                'support': loss_sup_v.item(),
                'smooth':  loss_smo_v.item(),
                'offset':  loss_off_v.item(),
                'rel_valid': loss_rel_valid_v.item(),
                'rel_invalid': loss_rel_invalid_v.item(),
                'temp': loss_temp_v.item(),
            }, glob_iter)
            writer.add_scalars('LossWeights', {
                'align': align_weight_v.item(),
                'triplet': triplet_weight_v.item(),
            }, glob_iter)
            writer.add_scalars('TripletDiagnostics', {
                'weight': triplet_weight_v.item(),
                'margin': out['triplet_margin'].mean().item(),
                'd_pos': triplet_d_pos_v.item(),
                'd_neg': triplet_d_neg_v.item(),
                'gap': triplet_gap_v.item(),
                'margin_gap': triplet_margin_gap_v.item(),
                'active_ratio': triplet_active_v.item(),
            }, glob_iter)
            writer.add_scalars('ConsensusDiagnostics', {
                'q_mean': q_mean_v.item(),
                'q_std': q_std_v.item(),
            }, glob_iter)
            writer.add_scalars('SigmaDiagnostics', {
                'sigma_mean': sigma_mean_v.item(),
                'log_sigma_mean': log_sigma_mean_v.item(),
                'log_sigma_min': log_sigma_min_v.item(),
                'log_sigma_max': log_sigma_max_v.item(),
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
    parser.add_argument('--lambda_align', type=float, default=1.0,
                        help='Final weight for the heteroscedastic alignment NLL.')
    parser.add_argument('--align_start_iters', type=int, default=4000,
                        help='Keep alignment NLL disabled before this iteration.')
    parser.add_argument('--align_warmup_iters', type=int, default=4000,
                        help='Linearly ramp lambda_align after align_start_iters.')
    parser.add_argument('--triplet_margin', type=float, default=0.05,
                        help='Triplet hinge margin matched to normalized v2 feature scale.')
    parser.add_argument('--lambda_triplet', type=float, default=1.0,
                        help='Final triplet loss weight for v1-style adaptation.')
    parser.add_argument('--triplet_start_iters', type=int, default=0,
                        help='Keep triplet disabled before this iteration.')
    parser.add_argument('--triplet_warmup_iters', type=int, default=0,
                        help='Linearly ramp lambda_triplet after triplet_start_iters.')
    parser.add_argument('--lambda_support', type=float, default=0.01)
    parser.add_argument('--lambda_smooth', type=float, default=0.001)
    parser.add_argument('--lambda_offset', type=float, default=1e-4)
    parser.add_argument('--lambda_rel', type=float, default=0.1)
    parser.add_argument('--fc_init_std', type=float, default=1e-2,
                        help='Std of Gaussian init on the geometry head fc. '
                             'Set 0.0 to recover the old zero-init (do not '
                             'do this — see v2 review §2a).')

    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    train(args)
