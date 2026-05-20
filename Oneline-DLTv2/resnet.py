"""
Calibrated Dominant-Plane Consensus Homography Estimation (CDPC).

Architecture additions over Oneline-DLTv1:
  ConsensusHead   – replaces genMask; outputs q_ab = P(z_i=1 | I_a, I_b)
  UncertaintyHead – predicts log_sigma_i (spatial uncertainty)
  ReliabilityHead – MLP predicting scalar s_ab = P(H reliable | I_a, I_b)
  Bidirectional   – both H_ab and H_ba via shared backbone, swapped concat
"""

import torch
import torch.nn as nn
from utils import transform, DLT_solve
from losses import (loss_align, loss_triplet, loss_inv,
                    loss_support, loss_smooth, loss_calib,
                    compute_total_loss)

__all__ = ['ResNetCDPC', 'resnet18_cdpc', 'resnet34_cdpc',
           'resnet50_cdpc', 'resnet101_cdpc']


# ---------------------------------------------------------------------------
# Shared building blocks (unchanged from v1)
# ---------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def getPatchFromFullimg(patch_size_h, patch_size_w,
                        patchIndices, batch_indices_tensor, img_full):
    """Extract a patch (1-channel) from a full image by flat index lookup."""
    num_batch = img_full.size(0)
    warped_images_flat = img_full.reshape(-1)
    patch_indices_flat = patchIndices.reshape(-1)
    pixel_indices = patch_indices_flat.long() + batch_indices_tensor
    patch = torch.gather(warped_images_flat, 0, pixel_indices)
    return patch.reshape([num_batch, 1, patch_size_h, patch_size_w])


def normMask(mask, strength=0.5):
    """Normalize mask so values are in [0, 1] with controllable saturation."""
    B = mask.size(0)
    max_val = mask.reshape(B, -1).max(1)[0].reshape(B, 1, 1, 1)
    return torch.clamp(mask / (max_val * strength + 1e-6), 0.0, 1.0)


def _build_conv_head(in_ch, mid_ch, out_ch, final_act):
    """Simple conv stack used for ConsensusHead and UncertaintyHead."""
    layers = [
        nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),

        nn.Conv2d(mid_ch, mid_ch * 2, 3, padding=1, bias=False),
        nn.BatchNorm2d(mid_ch * 2), nn.ReLU(inplace=True),

        nn.Conv2d(mid_ch * 2, mid_ch * 4, 3, padding=1, bias=False),
        nn.BatchNorm2d(mid_ch * 4), nn.ReLU(inplace=True),

        nn.Conv2d(mid_ch * 4, mid_ch * 2, 3, padding=1, bias=False),
        nn.BatchNorm2d(mid_ch * 2), nn.ReLU(inplace=True),

        nn.Conv2d(mid_ch * 2, out_ch, 3, padding=1, bias=False),
    ]
    if final_act == 'sigmoid':
        layers += [nn.BatchNorm2d(out_ch), nn.Sigmoid()]
    # 'none' → unbounded (log_sigma head)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ResNetCDPC(nn.Module):
    """
    Calibrated Dominant-Plane Consensus model.

    Outputs (forward dict):
        H_ab, H_ba               – bidirectional homographies  [B,3,3]
        offset_ab, offset_ba     – DLT corner offsets           [B,8]
        validity_prob_ab/ba      – consensus posterior q_i       [B,1,Ph,Pw]
        log_sigma_ab/ba          – spatial log-uncertainty       [B,1,Ph,Pw]
        reliability_score        – pair-level reliability s_ab   [B,1]
        residual_map_ab/ba       – |F_b - W(F_a,H)|             [B,1,Ph,Pw]
        loss_align/triplet/inv/support/smooth/calib/total
        pred_I2_d, patch_2_res_d, pred_F2_d, mask_ap_d          – visualisation
    """

    def __init__(self, block, layers, num_classes=8):
        self.inplanes = 64
        super().__init__()

        # --- Backbone (2-channel input: concat of masked features) ---
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # --- Shared feature extractor (1-ch in/out, for patch features) ---
        self.ShareFeature = nn.Sequential(
            nn.Conv2d(1, 4,  3, padding=1, bias=False), nn.BatchNorm2d(4),  nn.ReLU(inplace=True),
            nn.Conv2d(4, 8,  3, padding=1, bias=False), nn.BatchNorm2d(8),  nn.ReLU(inplace=True),
            nn.Conv2d(8, 1,  3, padding=1, bias=False), nn.BatchNorm2d(1),  nn.ReLU(inplace=True),
        )

        # --- Consensus posterior head: P(z_i=1 | I_a, I_b) ---
        # Takes 2-channel full image pair, outputs 1-channel sigmoid map.
        self.ConsensusHead = _build_conv_head(2, 8, 1, 'sigmoid')

        # --- Spatial uncertainty head: log sigma_i ---
        # Same topology as ConsensusHead; no final activation (unbounded).
        self.UncertaintyHead = _build_conv_head(2, 8, 1, 'none')

        # --- Reliability head: s_ab = P(H reliable | I_a, I_b) ---
        # Input: 7-dim feature vector phi_ab (see _reliability_features).
        self.ReliabilityHead = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # ------------------------------------------------------------------
    def _run_backbone(self, feat_a, feat_b):
        """Run the shared ResNet backbone on concatenated features."""
        x = torch.cat((feat_a, feat_b), dim=1)   # [B, 2, Ph, Pw]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)                            # [B, 8]
        return x

    def _reliability_features(self, q, r, log_sigma, H_ab, H_ba, tau=0.5):
        """
        Assemble the 7-dim reliability feature vector phi_ab (Section 5.6):
          [mean(q), var(q), area(q>tau), mean(r), mean(q*r),
           mean(sigma), ||H_ab H_ba - I||_F]
        """
        B = q.size(0)
        q_flat  = q.reshape(B, -1)
        r_flat  = r.reshape(B, -1)
        sig_flat = torch.exp(log_sigma).reshape(B, -1)

        q_mean   = q_flat.mean(1)
        q_var    = q_flat.var(1)
        q_area   = (q_flat > tau).float().mean(1)
        r_mean   = r_flat.mean(1)
        qr_mean  = (q_flat * r_flat).mean(1)
        s_mean   = sig_flat.mean(1)

        I3 = torch.eye(3, device=H_ab.device, dtype=H_ab.dtype) \
                   .unsqueeze(0).expand(B, -1, -1)
        inv_err = torch.norm(torch.bmm(H_ab, H_ba) - I3, dim=[1, 2])

        return torch.stack(
            [q_mean, q_var, q_area, r_mean, qr_mean, s_mean, inv_err], dim=1
        )   # [B, 7]

    # ------------------------------------------------------------------
    def forward(self, org_imges, input_tesnors, h4p, patch_indices):
        """
        org_imges    : [B, 2, H, W]   full image pair (normalised, grayscale)
        input_tesnors: [B, 2, Ph, Pw] patch pair
        h4p          : [B, 8]         source patch corners (flat)
        patch_indices: [B, Ph*Pw]     flat pixel indices for patch in full img
        """
        B, _, img_h, img_w = org_imges.size()
        _, _, Ph, Pw = input_tesnors.size()

        # ---- Shared coordinate setup --------------------------------
        y_t = torch.arange(0, B * img_w * img_h, img_w * img_h)
        batch_idx = y_t.unsqueeze(1).expand(B, Ph * Pw).reshape(-1)

        M = torch.tensor([[img_w / 2., 0., img_w / 2.],
                           [0., img_h / 2., img_h / 2.],
                           [0., 0., 1.]], dtype=org_imges.dtype)
        if org_imges.is_cuda:
            M = M.cuda()
            batch_idx = batch_idx.cuda()

        M_tile     = M.unsqueeze(0).expand(B, 3, 3)
        M_tile_inv = torch.inverse(M).unsqueeze(0).expand(B, 3, 3)

        # ---- 1. Patch feature extraction ---------------------------
        F1 = self.ShareFeature(input_tesnors[:, :1, ...])   # [B, 1, Ph, Pw]
        F2 = self.ShareFeature(input_tesnors[:, 1:, ...])   # [B, 1, Ph, Pw]

        # ---- 2. Consensus posterior (full img → patch region) -------
        # ConsensusHead receives 2-channel input (I_a, I_b).
        # For H_ba direction, channels are swapped.
        q_ab_full = self.ConsensusHead(org_imges)                        # [B,1,H,W]
        q_ba_full = self.ConsensusHead(org_imges[:, [1, 0], ...])        # swapped

        q_ab = getPatchFromFullimg(Ph, Pw, patch_indices, batch_idx, q_ab_full)
        q_ba = getPatchFromFullimg(Ph, Pw, patch_indices, batch_idx, q_ba_full)

        q_ab_w = normMask(q_ab)   # normalised for feature gating
        q_ba_w = normMask(q_ba)

        # ---- 3. Uncertainty head (full img → patch region) ----------
        log_s_ab_full = self.UncertaintyHead(org_imges)
        log_s_ba_full = self.UncertaintyHead(org_imges[:, [1, 0], ...])

        log_sigma_ab = getPatchFromFullimg(Ph, Pw, patch_indices, batch_idx, log_s_ab_full)
        log_sigma_ba = getPatchFromFullimg(Ph, Pw, patch_indices, batch_idx, log_s_ba_full)

        # ---- 4. Consensus-weighted features -------------------------
        # H_ab: source = I_a (F1), target = I_b (F2)
        F1_ab = torch.mul(F1, q_ab_w)
        F2_ab = torch.mul(F2, q_ab_w)
        # H_ba: source = I_b (F2), target = I_a (F1) – swapped concat
        F2_ba = torch.mul(F2, q_ba_w)
        F1_ba = torch.mul(F1, q_ba_w)

        # ---- 5. Backbone → offsets → homographies -------------------
        offset_ab = self._run_backbone(F1_ab, F2_ab)   # [B, 8]
        offset_ba = self._run_backbone(F2_ba, F1_ba)   # [B, 8]  (swapped)

        H_ab = DLT_solve(h4p, offset_ab).squeeze(1)   # [B, 3, 3]
        H_ba = DLT_solve(h4p, offset_ba).squeeze(1)   # [B, 3, 3]

        # ---- 6. Warp and residuals ----------------------------------
        # Warp I_a → I_b space using H_ab; extract patch, compute features
        pred_I2 = transform(Ph, Pw, M_tile_inv, H_ab, M_tile,
                            org_imges[:, :1, ...],
                            patch_indices, batch_idx)           # [B,1,Ph,Pw]
        pred_F2 = self.ShareFeature(pred_I2)

        # Warp I_b → I_a space using H_ba
        pred_I1 = transform(Ph, Pw, M_tile_inv, H_ba, M_tile,
                            org_imges[:, 1:, ...],
                            patch_indices, batch_idx)           # [B,1,Ph,Pw]
        pred_F1 = self.ShareFeature(pred_I1)

        r_ab = torch.abs(F2 - pred_F2)    # |F_b - W(F_a, H_ab)|
        r_ba = torch.abs(F1 - pred_F1)    # |F_a - W(F_b, H_ba)|

        # d_neg: feature distance without alignment
        d_neg_ab = torch.abs(F2 - F1)
        d_neg_ba = torch.abs(F1 - F2)

        # ---- 7. Reliability score ----------------------------------
        phi = self._reliability_features(q_ab, r_ab, log_sigma_ab, H_ab, H_ba)
        s_ab = self.ReliabilityHead(phi)   # [B, 1]

        # ---- 8. Losses ---------------------------------------------
        img_patch_b = input_tesnors[:, 1:, ...]   # for edge weights in L_smooth
        img_patch_a = input_tesnors[:, :1, ...]

        la  = loss_align(r_ab, log_sigma_ab, q_ab,
                         r_ba, log_sigma_ba, q_ba)
        lt  = loss_triplet(r_ab, d_neg_ab, q_ab,
                           r_ba, d_neg_ba, q_ba)
        li  = loss_inv(H_ab, H_ba)
        ls  = loss_support(q_ab, q_ba)
        lsm = loss_smooth(q_ab, q_ba, img_patch_b, img_patch_a)
        lc  = loss_calib(log_sigma_ab, r_ab, log_sigma_ba, r_ba)
        lt_total = compute_total_loss(la, lt, li, ls, lsm, lc)

        # ---- 9. Output dict ----------------------------------------
        return {
            # Homographies
            'H_ab': H_ab,
            'H_ba': H_ba,
            'H_mat': H_ab,          # backward-compat alias for test.py
            'offset_ab': offset_ab,
            'offset_ba': offset_ba,
            # Probabilistic outputs
            'validity_prob_ab': q_ab,
            'validity_prob_ba': q_ba,
            'log_sigma_ab': log_sigma_ab,
            'log_sigma_ba': log_sigma_ba,
            'reliability_score': s_ab,
            # Residuals
            'residual_map_ab': r_ab,
            'residual_map_ba': r_ba,
            # Losses
            'loss_align':   la,
            'loss_triplet': lt,
            'loss_inv':     li,
            'loss_support': ls,
            'loss_smooth':  lsm,
            'loss_calib':   lc,
            'loss_total':   lt_total,
            # Visualisation (first sample only)
            'pred_I2_d':            pred_I2[:1, ...],
            'patch_2_res_d':        F2_ab[:1, ...],
            'pred_F2_d':            pred_F2[:1, ...],
            'mask_ap_d':            q_ab[:1, 0, ...],
            'feature_loss':         lt_total.unsqueeze(0),   # compat
        }


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def resnet18_cdpc(**kwargs):
    return ResNetCDPC(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34_cdpc(**kwargs):
    return ResNetCDPC(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50_cdpc(**kwargs):
    return ResNetCDPC(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101_cdpc(**kwargs):
    return ResNetCDPC(Bottleneck, [3, 4, 23, 3], **kwargs)
