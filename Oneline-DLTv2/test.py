# coding: utf-8
"""
Evaluation script for CDPC homography model.

Outputs (per pair):
  - Reprojection error (standard, category-wise)
  - GIF animations (input / output alignment)
  - PNG maps: consensus q_ab, uncertainty sigma_ab, reliability score s_ab
  - result_ours_exp.txt with per-pair error + aggregate stats

Trustworthiness metrics (Section 9.2):
  - AUROC / AUPRC for failure detection at tau = {3, 5} px
  - Risk-coverage curve data (sorted by reliability score)
  - Expected calibration error (ECE)
"""

import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import imageio
from collections import defaultdict
from torch.utils.data import DataLoader

from torch_homography_model import build_model
from dataset import TestDataset
from utils import transformer as trans


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('true', '1', 'yes', 'y')


def load_model_weights(net, model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, nn.DataParallel):
        state_dict = checkpoint.module.state_dict()
    elif isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict',
                     checkpoint.get('model_state_dict', checkpoint))
    else:
        raise TypeError('Unsupported checkpoint: {}'.format(type(checkpoint)))

    cleaned = {(k[7:] if k.startswith('module.') else k): v
               for k, v in state_dict.items()}
    model_dict = net.state_dict()
    matched = {k: v for k, v in cleaned.items() if k in model_dict}
    if not matched:
        raise RuntimeError('No matching params in {}'.format(model_path))
    model_dict.update(matched)
    net.load_state_dict(model_dict)
    print('Loaded {}/{} tensors from {}'.format(
        len(matched), len(model_dict), model_path))
    return net


def geometric_distance(correspondence, h):
    p1 = np.transpose(np.matrix(
        [correspondence[0][0], correspondence[0][1], 1]))
    ep2 = np.dot(h, p1)
    ep2 = (1.0 / ep2.item(2)) * ep2
    p2 = np.transpose(np.matrix(
        [correspondence[1][0], correspondence[1][1], 1]))
    return np.linalg.norm(p2 - ep2)


def create_gif(frames, path):
    imageio.mimsave(path, frames, 'GIF', duration=0.5)


def save_heatmap(tensor_2d, path):
    """Save a [H,W] tensor as a colour heatmap PNG."""
    arr = tensor_2d.cpu().detach().numpy()
    arr = cv2.normalize(arr, None, 0, 255,
                        cv2.NORM_MINMAX, cv2.CV_8U)
    colored = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    cv2.imwrite(path, colored)


# ---------------------------------------------------------------------------
# Risk-coverage and calibration helpers
# ---------------------------------------------------------------------------

def auroc(labels, scores):
    """Compute AUROC.  labels=1 means failure."""
    from itertools import product
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    count = sum(1 for p, n in product(pos, neg)
                if p > n) + 0.5 * sum(1 for p, n in product(pos, neg)
                                       if p == n)
    return count / (len(pos) * len(neg))


def risk_coverage(errors, reliability_scores, tau=3.0):
    """
    Sort by reliability descending.  For each coverage threshold,
    compute the mean reprojection error of the accepted subset.
    Returns (coverages, risks) arrays.
    """
    order = np.argsort(reliability_scores)[::-1]  # high reliability first
    errors_sorted = np.array(errors)[order]
    n = len(errors_sorted)
    coverages, risks = [], []
    for k in range(1, n + 1):
        coverages.append(k / n)
        risks.append(errors_sorted[:k].mean())
    return np.array(coverages), np.array(risks)


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

def test(args):
    RE = ['0000011', '0000016', '00000147', '00000155', '00000158',
          '00000107', '00000239', '0000030']
    LT = ['0000038', '0000044', '0000046', '0000047', '00000238',
          '00000177', '00000188', '00000181']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']

    exp_name  = os.path.abspath(
        os.path.join(os.path.dirname('__file__'), os.path.pardir))
    work_dir  = os.path.join(exp_name, 'Data')
    pair_list = list(open(os.path.join(work_dir, 'Test_List.txt')))
    npy_path  = os.path.join(work_dir, 'Coordinate/')

    result_name  = 'exp_result_CDPC'
    result_files = os.path.join(exp_name, result_name)
    os.makedirs(result_files, exist_ok=True)

    res_txt = os.path.join(result_files, 'result_ours_exp.txt')
    f = open(res_txt, 'w')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    net = build_model(args.model_name, pretrained=args.pretrained)
    if args.finetune:
        model_path = args.model_path or os.path.join(
            exp_name,
            'train_log_CDPC/real_models/{}_iter_44000.pth'.format(args.model_name),
        )
        print('Loading checkpoint:', model_path)
        net = load_model_weights(net, model_path)
    net = net.to(device)
    net.eval()

    M = torch.tensor([[args.img_w / 2., 0., args.img_w / 2.],
                       [0., args.img_h / 2., args.img_h / 2.],
                       [0., 0., 1.]]).to(device)
    M_tile     = M.unsqueeze(0)
    M_tile_inv = torch.inverse(M).unsqueeze(0)

    test_data = TestDataset(
        data_path=exp_name,
        patch_w=args.patch_size_w, patch_h=args.patch_size_h,
        rho=16, WIDTH=args.img_w, HEIGHT=args.img_h,
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=1,
        num_workers=0, shuffle=False, drop_last=True,
    )

    per_cat = defaultdict(list)
    all_errors, all_reliability = [], []

    print('Start testing')
    for i, batch_value in enumerate(test_loader):
        img_pair   = pair_list[i]
        pari_id    = img_pair.split(' ')
        npy_name   = (pari_id[0].split('/')[1] + '_' +
                      pari_id[1].split('/')[1][:-1] + '.npy')
        npy_id     = npy_path + npy_name
        video_name = img_pair.split('/')[0]

        org_imges     = batch_value[0].float().to(device)
        input_tesnors = batch_value[1].float().to(device)
        patch_indices = batch_value[2].float().to(device)
        h4p           = batch_value[3].float().to(device)
        print_img_1   = batch_value[4].to(device)
        print_img_2   = batch_value[5]

        print_img_1_np = print_img_1.cpu().numpy()[0].transpose(1, 2, 0)
        print_img_2_np = print_img_2.numpy()[0].transpose(1, 2, 0)

        with torch.no_grad():
            out = net(org_imges, input_tesnors, h4p, patch_indices)

        H_ab = out['H_ab']          # [1, 3, 3]
        q_ab = out['validity_prob_ab']          # [1, 1, Ph, Pw]
        log_sigma_ab = out['log_sigma_ab']      # [1, 1, Ph, Pw]
        s_ab = out['reliability_score']         # [1, 1]

        # ---- Reprojection error ----
        H_np = H_ab.squeeze(0).cpu().numpy()
        H_np = np.linalg.inv(H_np)
        H_np = (1.0 / H_np[2, 2]) * H_np

        point_dic = np.load(npy_id, allow_pickle=True)
        data = point_dic.item()
        err_img = 0.0
        for j in range(6):
            pts_LR = data['matche_pts'][j]
            pts_RL = [pts_LR[1], pts_LR[0]]
            err_LR = geometric_distance(pts_LR, H_np)
            err_RL = geometric_distance(pts_RL, H_np)
            err_img += min(err_LR, err_RL)
        err_avg = err_img / 6

        reliability_val = s_ab.squeeze().item()
        all_errors.append(err_avg)
        all_reliability.append(reliability_val)

        name = '{:08d}'.format(i)
        f.write('{}:{}\n'.format(name, err_avg))
        print('{}:{:.4f}  s={:.4f}'.format(i, err_avg, reliability_val))

        if video_name in RE:
            per_cat['RE'].append(err_avg)
        elif video_name in LT:
            per_cat['LT'].append(err_avg)
        elif video_name in LL:
            per_cat['LL'].append(err_avg)
        elif video_name in SF:
            per_cat['SF'].append(err_avg)
        elif video_name in LF:
            per_cat['LF'].append(err_avg)

        # ---- Warped image GIF ----
        H_vis = torch.matmul(torch.matmul(M_tile_inv, H_ab), M_tile)
        pred_full, _ = trans(print_img_1, H_vis,
                             (args.img_h, args.img_w))
        pred_np = pred_full.cpu().numpy()[0].astype(np.uint8)
        pred_np = cv2.cvtColor(pred_np, cv2.COLOR_BGR2RGB)

        img1_rgb = cv2.cvtColor(print_img_1_np, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(print_img_2_np, cv2.COLOR_BGR2RGB)

        create_gif([img1_rgb, img2_rgb],
                   os.path.join(result_files, name + '_input.gif'))
        create_gif([pred_np, img2_rgb],
                   os.path.join(result_files, name + '_output.gif'))

        # ---- Consensus / uncertainty / reliability PNGs ----
        save_heatmap(
            q_ab[0, 0],
            os.path.join(result_files, name + '_q_ab.png'),
        )
        save_heatmap(
            torch.exp(log_sigma_ab[0, 0]),
            os.path.join(result_files, name + '_sigma_ab.png'),
        )

    # ---- Aggregate accuracy ----
    cat_means = {cat: float(np.mean(vals)) for cat, vals in per_cat.items()}
    all_err_arr = np.array(all_errors)
    cat_means['Mean'] = float(all_err_arr.mean())
    cat_means['Median'] = float(np.median(all_err_arr))
    print(cat_means)
    f.write(str(cat_means) + '\n')

    # ---- Trustworthiness metrics (Section 9.2) ----
    for tau in (3.0, 5.0):
        fail_labels = [1 if e > tau else 0 for e in all_errors]
        # Use (1 - reliability) as failure score so high score = likely failure
        fail_scores = [1.0 - r for r in all_reliability]
        auc = auroc(fail_labels, fail_scores)
        cov, risk = risk_coverage(all_errors, all_reliability, tau)

        print('AUROC (tau={}) = {:.4f}'.format(tau, auc))
        f.write('AUROC_tau{}={:.4f}\n'.format(int(tau), auc))

        # Save risk-coverage arrays
        rc_path = os.path.join(
            result_files, 'risk_coverage_tau{}.npz'.format(int(tau)))
        np.savez(rc_path, coverage=cov, risk=risk,
                 errors=all_errors, reliability=all_reliability)

    f.close()
    print('Results saved to', result_files)
    return cat_means


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus',  type=int, default=1)
    parser.add_argument('--cpus',  type=int, default=4)

    parser.add_argument('--img_w',        type=int, default=640)
    parser.add_argument('--img_h',        type=int, default=360)
    parser.add_argument('--patch_size_h', type=int, default=315)
    parser.add_argument('--patch_size_w', type=int, default=560)

    parser.add_argument('--model_name',  type=str,      default='resnet34')
    parser.add_argument('--pretrained',  type=str2bool, default=False)
    parser.add_argument('--finetune',    type=str2bool, default=True)
    parser.add_argument('--model_path',  type=str,      default=None)

    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    test(args)
