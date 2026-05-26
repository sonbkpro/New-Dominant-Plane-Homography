"""Multi-stage training driver for CDPC-v4.

Stages (4 total + a `full` end-to-end ablation):

  --stage synth : supervised geometric bootstrap on synthetic warps
                  Active: trunk + refiner.
                  Loss:   Huber(offset_total, offset_gt)
                          + Frobenius(H_full, H_full_gt)
                          + fold(offset_total)

  --stage geom  : unsupervised H training on real pairs
                  Active: trunk + refiner.
                  Loss:   triplet(sf_b, sf_a_warped, sf_a, valid_mask_patch)
                          + photo_img(I_a_full, I_b_patch, H_full)
                          + fold(offset_total)
                  This is the bulk-of-training stage; expect 50-150k iters
                  to reach the v1-target neighborhood (<0.5 px).

  --stage cdpc  : downstream posterior + uncertainty training
                  Active: posterior_head + uncertainty_head.
                  Loss:   em(q, r, sigma) + align_het(r, log_sigma, q)
                          + sigma_prior(log_sigma, r)
                          + support(q) + smooth(q, I_b_patch)
                  Geometry is frozen so q/sigma cannot collapse it.

  --stage rel   : reliability calibrator only
                  Active: reliability_head.
                  Loss:   BCE(s, invalid-pair label) on detached phi.

  --stage full  : end-to-end ablation (all modules train jointly).
                  Uses all geom and cdpc losses + rel.

Each stage logs the same panel: total loss + per-loss components + a
diagnostic block (offset norms, q/sigma stats, residual median, kappa).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _make_summary_writer_cls():
    try:
        from torch.utils.tensorboard import SummaryWriter as _SW
        return _SW
    except Exception as e:
        print(f"[warn] torch.utils.tensorboard unavailable ({type(e).__name__}); "
              f"trying tensorboardX", flush=True)
    try:
        from tensorboardX import SummaryWriter as _SW
        return _SW
    except Exception as e:
        print(f"[warn] tensorboardX unavailable ({type(e).__name__}); "
              f"disabling tensorboard logging (stdout only). "
              f"pip install tensorboardX to enable.", flush=True)

    class _NoOpWriter:
        def __init__(self, log_dir=None): self.log_dir = log_dir
        def add_scalar(self, *a, **kw): pass
        def add_image(self, *a, **kw): pass
        def close(self): pass

    return _NoOpWriter


SummaryWriter = _make_summary_writer_cls()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.path.pardir))
sys.path.insert(0, _THIS_DIR)

from configs.default import Config
from model.cdpc_net import CDPCNet
from losses.triplet import triplet_loss
from losses.align_v3 import align_het_loss, selection_set_stats
from losses.em import em_posterior_loss
from losses.support import support_loss
from losses.smooth import edge_aware_smoothness
from losses.reliability import reliability_loss, build_invalid_pair_labels
from losses.cycle import cycle_loss
from losses.fold import fold_loss
from losses.photo_image import photometric_image_loss
from losses.sigma_prior import sigma_prior_loss
from data.pairs import TrainPairDataset
from data.synth_pairs import SynthPairDataset
from data.synth_pairs_v3 import SynthPairDatasetV3
from data.invalid_pairs import build_invalid_batch
from data.test_dataset import TestDataset
from utils.eval_metrics import point_reprojection_error
from utils.inverse import safe_inverse_3x3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def patch_to_full_homography(H_patch: torch.Tensor, crop_xy: torch.Tensor) -> torch.Tensor:
    """H_full = T(+crop) @ H_patch @ T(-crop)."""
    B = H_patch.shape[0]
    device, dtype = H_patch.device, H_patch.dtype
    T = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, 0, 2] = crop_xy[:, 0]
    T[:, 1, 2] = crop_xy[:, 1]
    T_inv = T.clone()
    T_inv[:, 0, 2] = -crop_xy[:, 0]
    T_inv[:, 1, 2] = -crop_xy[:, 1]
    return torch.bmm(torch.bmm(T, H_patch), T_inv)


def _eye_like(H: torch.Tensor) -> torch.Tensor:
    B = H.shape[0]
    I = torch.eye(3, device=H.device, dtype=H.dtype)
    return I.unsqueeze(0).expand(B, -1, -1).contiguous()


def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


# Module names in CDPC-v4. The geometry trunks (trunk + refiner) form one
# logical group for staged freezing; the CDPC heads form another.
def _freeze_for_stage(net, stage: str):
    all_mods = {
        "trunk":            net.trunk,
        "refiner":          net.refiner,
        "posterior_head":   net.posterior_head,
        "uncertainty_head": net.uncertainty_head,
        "reliability_head": net.reliability_head,
    }
    if stage == "synth":
        active = {"trunk", "refiner"}
    elif stage == "geom":
        active = {"trunk", "refiner"}
    elif stage == "cdpc":
        active = {"posterior_head", "uncertainty_head"}
    elif stage == "rel":
        active = {"reliability_head"}
    elif stage == "full":
        active = set(all_mods.keys())
    else:
        raise ValueError(
            f"unknown --stage: {stage}. "
            f"Valid: synth, geom, cdpc, rel, full."
        )
    for name, mod in all_mods.items():
        _set_requires_grad(mod, name in active)
    return active


def _make_param_groups(net, cfg: Config, stage: str):  # noqa: ARG001 (stage kept for symmetry with v3 callers)
    """Differential LRs per group. Only trainable params end up in the
    optimizer so frozen modules contribute zero-size groups (excluded)."""
    trunk_mods   = [net.trunk]
    refiner_mods = [net.refiner]
    head_mods    = [net.posterior_head, net.uncertainty_head, net.reliability_head]

    def _params(mods):
        ps = []
        for m in mods:
            ps += [p for p in m.parameters() if p.requires_grad]
        return ps

    lr_trunk   = cfg.lr_trunk
    lr_refiner = cfg.lr_refiner
    lr_heads   = cfg.lr_heads

    groups = []
    p_t = _params(trunk_mods)
    p_r = _params(refiner_mods)
    p_h = _params(head_mods)
    if p_t:
        groups.append({"params": p_t, "lr": lr_trunk})
    if p_r:
        groups.append({"params": p_r, "lr": lr_refiner})
    if p_h:
        groups.append({"params": p_h, "lr": lr_heads})
    if not groups:
        groups.append({"params": list(net.parameters()), "lr": cfg.lr})
    return groups


def _save_checkpoint(net, save_dir: str, filename: str, **extra) -> str:
    path = os.path.join(save_dir, filename)
    try:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), **extra}, path)
        print(f"[ckpt] saved {path}", flush=True)
        return path
    except (OSError, RuntimeError) as e:
        print(f"[ckpt] FAILED to save {path}: {type(e).__name__}: {e}", flush=True)
        return ""


def _try_load_init(net, ckpt_path: str):
    """Soft-load (strict=False) a checkpoint when explicitly requested."""
    if not ckpt_path:
        return
    candidates = [ckpt_path]
    if not os.path.isabs(ckpt_path):
        candidates.append(os.path.normpath(os.path.join(_REPO_ROOT, ckpt_path)))
    resolved = next((c for c in candidates if os.path.isfile(c)), None)
    if resolved is None:
        raise FileNotFoundError(
            f"--init_ckpt requested but the file was not found.\n"
            f"  Looked at: {candidates}"
        )
    ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[init_ckpt] missing  ({len(missing)}): {missing[:6]}...", flush=True)
    if unexpected:
        print(f"[init_ckpt] unexpected ({len(unexpected)}): {unexpected[:6]}...", flush=True)
    print(f"[init_ckpt] loaded {resolved}", flush=True)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

SCENES = ("RE", "LT", "LL", "SF", "LF")
EVAL_METRICS = ("direct", "inverse", "symmetric", "identity", "v1", "v1_compat")


@torch.no_grad()
def run_eval_l2(net, test_loader, device, max_batches=None) -> dict:
    net.eval()
    per_scene = {m: {s: [] for s in SCENES} for m in EVAL_METRICS}

    for i, batch in enumerate(test_loader):
        if max_batches is not None and i >= max_batches:
            break
        I_a_full  = batch["I_a_full"].to(device, non_blocking=True)
        I_b_full  = batch["I_b_full"].to(device, non_blocking=True)
        I_a_patch = batch["I_a_patch"].to(device, non_blocking=True)
        I_b_patch = batch["I_b_patch"].to(device, non_blocking=True)
        crop_xy   = batch["crop_xy"].to(device, non_blocking=True)

        out = net(I_a_full, I_b_full, I_a_patch, I_b_patch, crop_xy)
        H_full = patch_to_full_homography(out["H_ab"], crop_xy)
        H_full_inv, _ = safe_inverse_3x3(H_full)

        pts = batch["points"].to(device)[:, :6, :, :]
        pts_a = pts[:, :, 0, :]
        pts_b = pts[:, :, 1, :]

        err_ab      = point_reprojection_error(H_full,            pts_a, pts_b)
        err_ba      = point_reprojection_error(H_full,            pts_b, pts_a)
        err_inv     = point_reprojection_error(H_full_inv,        pts_b, pts_a)
        err_inv_alt = point_reprojection_error(H_full_inv,        pts_a, pts_b)
        err_id      = point_reprojection_error(_eye_like(H_full), pts_a, pts_b)

        per_pair = {
            "direct":    err_ab.mean(dim=1),
            "inverse":   err_inv.mean(dim=1),
            "symmetric": ((err_ab + err_inv) / 2.0).mean(dim=1),
            "identity":  err_id.mean(dim=1),
            "v1":        torch.minimum(err_ab, err_ba).mean(dim=1),
            "v1_compat": torch.minimum(err_inv, err_inv_alt).mean(dim=1),
        }
        for j in range(I_a_patch.shape[0]):
            scene = batch["scene"][j]
            if scene in SCENES:
                for m in EVAL_METRICS:
                    per_scene[m][scene].append(float(per_pair[m][j]))

    net.train()

    summary = {}
    for m in EVAL_METRICS:
        scene_means = {s: float(np.mean(per_scene[m][s])) if per_scene[m][s] else float("nan")
                       for s in SCENES}
        vals = [scene_means[s] for s in SCENES if not np.isnan(scene_means[s])]
        avg = float(np.mean(vals)) if vals else float("nan")
        summary[m] = {**scene_means, "overall": avg}
    return summary


def _ramp(it, warmup, ramp):
    if it < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, (it - warmup) / float(ramp))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args, cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage = cfg.stage

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)

    # --- model ---
    net = CDPCNet(
        patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        trunk_pretrained=cfg.trunk_pretrained,
        trunk_rho_init=cfg.trunk_rho_init,
        share_feature_channels=cfg.share_feature_channels,
        refiner_feat_channels=cfg.refiner_feat_channels,
        refiner_radius=cfg.refiner_radius,
        refiner_pretrained=cfg.refiner_pretrained,
        refiner_corner_window=cfg.refiner_corner_window,
        refiner_temperature=cfg.refiner_temperature,
        refiner_rho_max=cfg.refiner_rho_max,
        use_normalized_dlt=cfg.use_normalized_dlt,
        post_init_prob=cfg.post_init_prob,
        sigma_min=cfg.sigma_min,
    ).to(device)
    _try_load_init(net, cfg.init_ckpt)
    active = _freeze_for_stage(net, stage)
    print(f"[stage={stage}] active modules: {sorted(active)}", flush=True)
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    n_total = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"[stage={stage}] trainable: {n_train:.2f}M / {n_total:.2f}M", flush=True)

    # --- data ---
    if stage == "synth":
        SynthCls = SynthPairDatasetV3 if cfg.use_v3_synth else SynthPairDataset
        train_ds = SynthCls(
            _abs(cfg.train_list), _abs(cfg.train_root),
            patch_h=cfg.patch_h, patch_w=cfg.patch_w,
            img_h=cfg.img_h, img_w=cfg.img_w, rho=cfg.rho,
            rho_s=cfg.synth_rho_max,
        )
    else:
        train_ds = TrainPairDataset(
            _abs(cfg.train_list), _abs(cfg.train_root),
            patch_h=cfg.patch_h, patch_w=cfg.patch_w,
            img_h=cfg.img_h, img_w=cfg.img_w, rho=cfg.rho,
            max_frame_gap=getattr(args, "max_frame_gap", 0),
        )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, num_workers=args.cpus,
        shuffle=True, drop_last=True, pin_memory=True,
    )

    test_ds = TestDataset(
        data_root=_REPO_ROOT,
        patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        img_h=cfg.img_h, img_w=cfg.img_w,
    )
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0, shuffle=False)
    print(f"[train] {len(train_ds)} train, {len(test_ds)} test", flush=True)

    # --- optimizer ---
    param_groups = _make_param_groups(net, cfg, stage)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    if cfg.use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epoch, eta_min=1e-7,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    # --- logging ---
    log_dir  = _abs(cfg.log_dir)
    save_dir = _abs(cfg.model_save_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[train] logging to {log_dir}", flush=True)
    print(f"[train] checkpoints to {save_dir}", flush=True)

    glob_iter = 0
    for epoch in range(cfg.max_epoch):
        net.train()
        for batch_idx, batch in enumerate(train_loader):
            I_a_full  = batch["I_a_full"].to(device, non_blocking=True)
            I_b_full  = batch["I_b_full"].to(device, non_blocking=True)
            I_a_patch = batch["I_a_patch"].to(device, non_blocking=True)
            I_b_patch = batch["I_b_patch"].to(device, non_blocking=True)
            crop_xy   = batch["crop_xy"].to(device, non_blocking=True)
            B = I_a_patch.shape[0]

            # Invalid-pair construction only matters when reliability trains.
            do_rel = (stage in ("rel", "full")) and (glob_iter >= cfg.rel_warmup_iters)
            if do_rel:
                I_a_neg, I_b_neg, y_neg = build_invalid_batch(
                    I_a_patch, I_b_patch,
                    shuffle_frac=cfg.rel_shuffle_frac,
                    reshuffle_frac=cfg.rel_reshuffle_frac,
                )
                I_a_full_neg = I_a_full[:I_a_neg.shape[0]]
                I_b_full_neg = I_b_full[:I_b_neg.shape[0]]
                crop_xy_neg = crop_xy[:I_a_neg.shape[0]]
                I_a_full_all  = torch.cat([I_a_full,  I_a_full_neg], dim=0)
                I_b_full_all  = torch.cat([I_b_full,  I_b_full_neg], dim=0)
                I_a_patch_all = torch.cat([I_a_patch, I_a_neg], dim=0)
                I_b_patch_all = torch.cat([I_b_patch, I_b_neg], dim=0)
                crop_xy_all   = torch.cat([crop_xy,  crop_xy_neg], dim=0)
            else:
                I_a_full_all  = I_a_full
                I_b_full_all  = I_b_full
                I_a_patch_all = I_a_patch
                I_b_patch_all = I_b_patch
                crop_xy_all   = crop_xy
                y_neg = None

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                out = net(I_a_full_all, I_b_full_all,
                          I_a_patch_all, I_b_patch_all, crop_xy_all)

            # All losses computed in fp32 to keep DLT / linalg stable under AMP.
            with torch.amp.autocast("cuda", enabled=False):
                # ShareFeature triplet inputs (patch resolution, 1-ch).
                sf_b           = out["sf_b"][:B].float()
                sf_a_warped    = out["sf_a_warped"][:B].float()
                sf_a           = out["sf_a"][:B].float()
                valid_patch    = out["valid_mask_patch"][:B].float()

                # 1/4-resolution refiner features (used by cdpc, cycle, fold).
                F_a4           = out["F_a4"][:B].float()
                F_a_rec        = out["F_a_recovered"][:B].float()
                q_nat          = out["q"][:B].float()
                log_sigma_nat  = out["log_sigma"][:B].float()
                residual_nat   = out["residual"][:B].float()
                valid_quarter  = out["valid_mask"][:B].float()
                cycle_valid    = out["cycle_valid"][:B].float()
                cond_valid     = out["cond_valid"][:B].float()
                offset_nat     = out["offset"][:B].float()
                H_patch_nat    = out["H_patch"][:B].float()
                s_all          = out["s"].float()

                I_a_full_nat  = I_a_full_all[:B].float()
                I_b_patch_nat = I_b_patch_all[:B].float()

                H_full_nat = patch_to_full_homography(H_patch_nat, crop_xy[:B].float())

                L = dict.fromkeys(
                    ["triplet", "photo_img", "fold", "sup_corner", "sup_H",
                     "em", "support", "smooth", "align_het", "sigma_prior",
                     "cycle", "rel"],
                    torch.zeros((), device=device),
                )

                # ---- Always-on: fold penalty when H trains ----
                if stage in ("synth", "geom", "full"):
                    L["fold"] = fold_loss(offset_nat, cfg.patch_h, cfg.patch_w)

                # ============================================================
                # SYNTH: supervised corner offset + Frobenius
                # ============================================================
                if stage == "synth":
                    if "offset_gt" not in batch:
                        raise KeyError(
                            "stage=synth requires offset_gt; the training "
                            "dataloader is not a SynthPair dataset.")
                    offset_gt = batch["offset_gt"].to(device).float()
                    L["sup_corner"] = F.smooth_l1_loss(offset_nat, offset_gt, beta=1.0)
                    if "H_full_gt" in batch:
                        H_full_gt = batch["H_full_gt"].to(device).float()
                        H_pred = H_full_nat / (H_full_nat[:, 2:3, 2:3] + 1e-8)
                        H_gt   = H_full_gt   / (H_full_gt[:, 2:3, 2:3]   + 1e-8)
                        denom = H_gt.flatten(1).norm(dim=1).clamp(min=1e-6)
                        diff  = (H_pred - H_gt).flatten(1).norm(dim=1)
                        L["sup_H"] = (diff / denom).mean()

                # ============================================================
                # GEOM: unsupervised real-pair training (triplet + photo_img)
                # ============================================================
                if stage in ("geom", "synth", "full"):
                    # Synth stage also gets the triplet + photo_img signal as
                    # an auxiliary unsupervised cue (cheap, doesn't hurt).
                    L["triplet"] = triplet_loss(
                        sf_b, sf_a_warped, sf_a,
                        q=valid_patch,  # use valid mask as the per-pixel weight (no q gating)
                        valid_mask=valid_patch,
                        margin=cfg.triplet_margin,
                        use_q_weighting=False,
                    )
                    L["photo_img"] = photometric_image_loss(
                        I_a_full_nat, I_b_patch_nat,
                        H_full_nat, crop_xy[:B].float(),
                    )

                # ============================================================
                # CDPC: posterior + uncertainty (downstream, geometry frozen)
                # ============================================================
                if stage in ("cdpc", "full"):
                    em_warmed = (stage == "cdpc") or (glob_iter >= cfg.em_warmup_iters)
                    if em_warmed:
                        beta = cfg.em_self_paced_beta if cfg.use_em_self_paced_gate else 0.0
                        L["em"] = em_posterior_loss(
                            q_nat, residual_nat, log_sigma_nat, valid_quarter,
                            pi=cfg.em_prior_pi, r_max=cfg.em_r_max,
                            self_paced_beta=beta,
                        )
                    L["support"] = support_loss(q_nat, valid_quarter, alpha=cfg.alpha_support)
                    L["smooth"]  = edge_aware_smoothness(
                        q_nat, I_b_patch_nat, gamma=cfg.smoothness_gamma,
                    )
                    L["align_het"] = align_het_loss(
                        residual_nat, log_sigma_nat, q_nat, valid_quarter,
                        tau_q=cfg.q_select_tau,
                    )
                    L["sigma_prior"] = sigma_prior_loss(
                        log_sigma_nat, residual_nat, valid_quarter,
                    )

                # Cycle loss is a diagnostic / safety prior on the refiner.
                # Only enabled in `full` stage; geometry-only stages should
                # not be slowed by it.
                if stage == "full":
                    L["cycle"] = cycle_loss(
                        F_a4, F_a_rec, cycle_valid, cond_valid,
                    )

                # ============================================================
                # REL: reliability calibrator on detached phi
                # ============================================================
                if do_rel:
                    r_mean_nat = residual_nat.mean(dim=(1, 2, 3))
                    y_nat = build_invalid_pair_labels(
                        r_mean_nat,
                        hard_negative_percentile=cfg.rel_hard_neg_percentile,
                    )
                    y_target = torch.cat([y_nat, y_neg.float()], dim=0)
                    L["rel"] = reliability_loss(s_all, y_target)

                # ---- Total ----
                lam_em_eff = cfg.lambda_em * _ramp(
                    glob_iter,
                    0 if stage == "cdpc" else cfg.em_warmup_iters,
                    cfg.em_ramp_iters,
                )
                L_total = (
                    cfg.lambda_triplet     * L["triplet"]
                    + cfg.lambda_photo_img * L["photo_img"]
                    + cfg.lambda_fold      * L["fold"]
                    + cfg.lambda_sup_corner * L["sup_corner"]
                    + cfg.lambda_sup_H     * L["sup_H"]
                    + lam_em_eff           * L["em"]
                    + cfg.lambda_align_het * L["align_het"]
                    + cfg.lambda_sigma     * L["sigma_prior"]
                    + cfg.lambda_support   * L["support"]
                    + cfg.lambda_smooth    * L["smooth"]
                    + cfg.lambda_cycle     * L["cycle"]
                    + cfg.lambda_rel       * L["rel"]
                )

            scaler.scale(L_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad],
                cfg.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()

            # --- diagnostics ---
            with torch.no_grad():
                offset_abs = offset_nat.abs()
                offset_inf = float(offset_abs.max())
                offset_mean = float(offset_abs.mean())
                offset_init_mean = float(out["offset_init"][:B].abs().mean())
                offset_resid_mean = float(out["offset_residual"][:B].abs().mean())
                kappa_H_nat = out["kappa_H"][:B].detach().float()
                kappa_log = float(torch.log(kappa_H_nat.clamp(min=1.0)).mean())
                fold_count = float((L["fold"] > 0).float())
                area_valid_q = float(valid_quarter.mean())
                area_valid_p = float(valid_patch.mean())
                q_mean = float(q_nat.mean())
                q_var  = float(((q_nat - q_nat.mean()) ** 2).mean())
                q_area = float((q_nat > 0.5).float().mean())
                sigma_mean = float(log_sigma_nat.exp().mean())
                r_median   = float(residual_nat.median())
                sel_frac, q_in_S = selection_set_stats(
                    q_nat, valid_quarter, tau_q=cfg.q_select_tau)
                sel_frac_m = float(sel_frac.mean())
                q_in_S_m   = float(q_in_S.mean())

            for k, v in L.items():
                writer.add_scalar(f"loss/{k}", float(v), glob_iter)
            writer.add_scalar("loss/total", float(L_total), glob_iter)
            writer.add_scalar("opt/lr", optimizer.param_groups[0]["lr"], glob_iter)
            writer.add_scalar("opt/lambda_em_eff", lam_em_eff, glob_iter)
            writer.add_scalar("H/offset_inf_px",  offset_inf,  glob_iter)
            writer.add_scalar("H/offset_mean_px", offset_mean, glob_iter)
            writer.add_scalar("H/offset_init_mean_px",  offset_init_mean, glob_iter)
            writer.add_scalar("H/offset_residual_mean_px", offset_resid_mean, glob_iter)
            writer.add_scalar("H/kappa_log",      kappa_log,   glob_iter)
            writer.add_scalar("H/fold_count",     fold_count,  glob_iter)
            writer.add_scalar("q/mean",           q_mean,      glob_iter)
            writer.add_scalar("q/var",            q_var,       glob_iter)
            writer.add_scalar("q/area_above_tau", q_area,      glob_iter)
            writer.add_scalar("q/A_v_quarter",    area_valid_q, glob_iter)
            writer.add_scalar("q/A_v_patch",      area_valid_p, glob_iter)
            writer.add_scalar("q/selected_frac",  sel_frac_m,  glob_iter)
            writer.add_scalar("q/in_S_mean",      q_in_S_m,    glob_iter)
            writer.add_scalar("sigma/mean",       sigma_mean,  glob_iter)
            writer.add_scalar("residual/median",  r_median,    glob_iter)

            if glob_iter % 100 == 0:
                for mod_name, mod in [
                    ("trunk",            net.trunk),
                    ("refiner",          net.refiner),
                    ("posterior_head",   net.posterior_head),
                    ("uncertainty_head", net.uncertainty_head),
                    ("reliability_head", net.reliability_head),
                ]:
                    g2 = 0.0; n = 0
                    for p in mod.parameters():
                        if p.grad is not None:
                            g2 += float(p.grad.norm() ** 2)
                            n += 1
                    if n > 0:
                        writer.add_scalar(f"grad_norm/{mod_name}", g2 ** 0.5, glob_iter)

            if batch_idx % cfg.score_print_freq == 0:
                msg = (f"[ep {epoch+1:02d} it {glob_iter:06d} stage={stage}] "
                       f"L={float(L_total):.4f}  "
                       f"sup_c={float(L['sup_corner']):.3f} "
                       f"sup_H={float(L['sup_H']):.3f} "
                       f"trip={float(L['triplet']):.3f} "
                       f"ph_img={float(L['photo_img']):.3f} "
                       f"em={float(L['em']):.3f} "
                       f"ahet={float(L['align_het']):.3f} "
                       f"sig={float(L['sigma_prior']):.3f} "
                       f"sup={float(L['support']):.3f} "
                       f"sm={float(L['smooth']):.4f} "
                       f"cyc={float(L['cycle']):.3f} "
                       f"rel={float(L['rel']):.3f} "
                       f"fold={float(L['fold']):.3f}  "
                       f"q={q_mean:.3f} sig_e={sigma_mean:.3f} "
                       f"r_med={r_median:.3f} "
                       f"off_init={offset_init_mean:.2f}px "
                       f"off_res={offset_resid_mean:.2f}px "
                       f"off_inf={offset_inf:.2f}px "
                       f"kap_log={kappa_log:.2f}")
                print(msg, flush=True)

            if glob_iter > 0 and glob_iter % cfg.eval_every == 0:
                t0 = time.time()
                summary = run_eval_l2(net, test_loader, device,
                                      max_batches=args.eval_max_batches)
                dt = time.time() - t0
                for m, scene_dict in summary.items():
                    for k, v in scene_dict.items():
                        writer.add_scalar(f"eval_l2/{m}/{k}", v, glob_iter)
                line = "  ".join(f"{m}={summary[m]['overall']:.3f}"
                                 for m in EVAL_METRICS)
                print(f"[eval @ it {glob_iter}] {line}  ({dt:.1f}s)", flush=True)
                # Spotlight the v1_compat per-scene numbers (paper targets).
                vc = summary["v1_compat"]
                tgts = ("RE", "LT", "LL", "SF", "LF")
                tline = "  ".join(f"{s}={vc[s]:.3f}" for s in tgts)
                print(f"[eval @ it {glob_iter}] v1_compat per-scene: {tline}"
                      f"  AVG={vc['overall']:.3f}", flush=True)

            if glob_iter > 0 and glob_iter % cfg.model_save_freq == 0:
                _save_checkpoint(
                    net, save_dir,
                    f"cdpc_{stage}_iter_{glob_iter}.pth",
                    iter=glob_iter, epoch=epoch, stage=stage,
                    config=cfg.__dict__,
                )

            glob_iter += 1

            if stage == "synth" and batch_idx >= cfg.synth_iters_per_epoch:
                break

        scheduler.step()

    final_path = _save_checkpoint(
        net, save_dir,
        f"cdpc_{stage}_iter_{glob_iter}_final.pth",
        iter=glob_iter, epoch=cfg.max_epoch, stage=stage,
        config=cfg.__dict__,
    )
    if final_path:
        print(f"[done] final checkpoint: {final_path}", flush=True)
    writer.close()


def _parse_args(cfg: Config):
    p = argparse.ArgumentParser()
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--max_epoch", type=int, default=cfg.max_epoch)
    p.add_argument("--lr",       type=float, default=cfg.lr)
    p.add_argument("--lr_trunk",   type=float, default=cfg.lr_trunk)
    p.add_argument("--lr_refiner", type=float, default=cfg.lr_refiner)
    p.add_argument("--lr_heads",   type=float, default=cfg.lr_heads)
    p.add_argument("--patch_h",  type=int, default=cfg.patch_h)
    p.add_argument("--patch_w",  type=int, default=cfg.patch_w)
    p.add_argument("--img_h",    type=int, default=cfg.img_h)
    p.add_argument("--img_w",    type=int, default=cfg.img_w)
    p.add_argument("--amp",      action="store_true", default=cfg.use_amp)
    p.add_argument("--no_amp",   dest="amp", action="store_false")
    p.add_argument("--eval_every", type=int, default=cfg.eval_every)
    p.add_argument("--eval_max_batches", type=int, default=None)
    p.add_argument("--train_list", type=str, default=cfg.train_list)
    p.add_argument("--train_root", type=str, default=cfg.train_root)
    p.add_argument("--log_dir",   type=str, default=cfg.log_dir)
    p.add_argument("--model_save_dir", type=str, default=cfg.model_save_dir)
    p.add_argument("--stage", type=str, default=cfg.stage,
                   choices=["synth", "geom", "cdpc", "rel", "full"])
    p.add_argument("--init_ckpt", type=str, default=cfg.init_ckpt)
    p.add_argument("--max_frame_gap", type=int, default=0)
    p.add_argument("--synth_rho_max", type=int, default=cfg.synth_rho_max)
    p.add_argument("--use_v3_synth", action="store_true", default=cfg.use_v3_synth)
    p.add_argument("--no_v3_synth", dest="use_v3_synth", action="store_false")
    p.add_argument("--model_save_freq", type=int, default=cfg.model_save_freq)
    # ---- Trunk / refiner ----
    p.add_argument("--trunk_pretrained", action="store_true",
                   default=cfg.trunk_pretrained)
    p.add_argument("--no_trunk_pretrained", dest="trunk_pretrained",
                   action="store_false")
    p.add_argument("--trunk_rho_init",     type=float, default=cfg.trunk_rho_init)
    p.add_argument("--refiner_feat_channels", type=int, default=cfg.refiner_feat_channels)
    p.add_argument("--refiner_radius",     type=int, default=cfg.refiner_radius)
    p.add_argument("--refiner_corner_window", type=int, default=cfg.refiner_corner_window)
    p.add_argument("--refiner_temperature",   type=float, default=cfg.refiner_temperature)
    p.add_argument("--refiner_rho_max",       type=float, default=cfg.refiner_rho_max)
    # ---- Loss weights ----
    p.add_argument("--lambda_triplet",    type=float, default=cfg.lambda_triplet)
    p.add_argument("--lambda_photo_img",  type=float, default=cfg.lambda_photo_img)
    p.add_argument("--lambda_fold",       type=float, default=cfg.lambda_fold)
    p.add_argument("--lambda_align_het",  type=float, default=cfg.lambda_align_het)
    p.add_argument("--lambda_em",         type=float, default=cfg.lambda_em)
    p.add_argument("--lambda_support",    type=float, default=cfg.lambda_support)
    p.add_argument("--lambda_smooth",     type=float, default=cfg.lambda_smooth)
    p.add_argument("--lambda_sigma",      type=float, default=cfg.lambda_sigma)
    p.add_argument("--lambda_cycle",      type=float, default=cfg.lambda_cycle)
    p.add_argument("--lambda_rel",        type=float, default=cfg.lambda_rel)
    p.add_argument("--lambda_sup_corner", type=float, default=cfg.lambda_sup_corner)
    p.add_argument("--lambda_sup_H",      type=float, default=cfg.lambda_sup_H)
    p.add_argument("--triplet_margin",    type=float, default=cfg.triplet_margin)
    p.add_argument("--sigma_min",         type=float, default=cfg.sigma_min)
    p.add_argument("--alpha_support",     type=float, default=cfg.alpha_support)
    # ---- Warmups (used in `full`) ----
    p.add_argument("--em_warmup_iters",    type=int, default=cfg.em_warmup_iters)
    p.add_argument("--em_ramp_iters",      type=int, default=cfg.em_ramp_iters)
    p.add_argument("--use_cosine_lr",      action="store_true", default=cfg.use_cosine_lr)
    p.add_argument("--no_cosine_lr",       dest="use_cosine_lr", action="store_false")
    args = p.parse_args()

    cfg.batch_size = args.batch_size
    cfg.max_epoch = args.max_epoch
    cfg.lr = args.lr
    cfg.lr_trunk = args.lr_trunk
    cfg.lr_refiner = args.lr_refiner
    cfg.lr_heads = args.lr_heads
    cfg.patch_h = args.patch_h
    cfg.patch_w = args.patch_w
    cfg.img_h = args.img_h
    cfg.img_w = args.img_w
    cfg.use_amp = args.amp
    cfg.eval_every = args.eval_every
    cfg.train_list = args.train_list
    cfg.train_root = args.train_root
    cfg.log_dir = args.log_dir
    cfg.model_save_dir = args.model_save_dir
    cfg.stage = args.stage
    cfg.init_ckpt = args.init_ckpt
    cfg.synth_rho_max = args.synth_rho_max
    cfg.use_v3_synth = args.use_v3_synth
    cfg.model_save_freq = args.model_save_freq
    cfg.trunk_pretrained = args.trunk_pretrained
    cfg.trunk_rho_init = args.trunk_rho_init
    cfg.refiner_feat_channels = args.refiner_feat_channels
    cfg.refiner_radius = args.refiner_radius
    cfg.refiner_corner_window = args.refiner_corner_window
    cfg.refiner_temperature = args.refiner_temperature
    cfg.refiner_rho_max = args.refiner_rho_max
    cfg.lambda_triplet = args.lambda_triplet
    cfg.lambda_photo_img = args.lambda_photo_img
    cfg.lambda_fold = args.lambda_fold
    cfg.lambda_align_het = args.lambda_align_het
    cfg.lambda_em = args.lambda_em
    cfg.lambda_support = args.lambda_support
    cfg.lambda_smooth = args.lambda_smooth
    cfg.lambda_sigma = args.lambda_sigma
    cfg.lambda_cycle = args.lambda_cycle
    cfg.lambda_rel = args.lambda_rel
    cfg.lambda_sup_corner = args.lambda_sup_corner
    cfg.lambda_sup_H = args.lambda_sup_H
    cfg.triplet_margin = args.triplet_margin
    cfg.sigma_min = args.sigma_min
    cfg.alpha_support = args.alpha_support
    cfg.em_warmup_iters = args.em_warmup_iters
    cfg.em_ramp_iters = args.em_ramp_iters
    cfg.use_cosine_lr = args.use_cosine_lr
    return args


if __name__ == "__main__":
    cfg = Config()
    args = _parse_args(cfg)
    print(f"[config] {cfg}", flush=True)
    train(args, cfg)
