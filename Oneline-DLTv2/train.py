"""Multi-stage training driver for CDPC v2.

Stages (planv2):

  --stage synth   : synthetic supervised H bootstrap.
                    Active: joint_backbone + homography_head.
                    Loss: Huber(offset_pred, offset_gt) + L_fold.
                    Frozen: per-image backbone, posterior, uncertainty, reliability.

  --stage h_only  : real-pair H-only.
                    Active: per-image backbone, joint_backbone, correlation,
                            homography_head.
                    Loss: triplet + photometric (Charbonnier) + smooth + L_fold.
                    Frozen: posterior, uncertainty, reliability.

  --stage q_sigma : add posterior + uncertainty.
                    Active: everything above + posterior_head + uncertainty_head.
                    Loss: + L_align (q-weighted) + L_em (residual-gated) + L_support.
                    Frozen: reliability.

  --stage joint   : joint fine-tune with q-dagger.
                    Active: all geometry + q + sigma.
                    Loss: L_align uses q_dagger = sg(max(q, q_min_floor)),
                          decoupling H learning from q transients.
                    Frozen: reliability. Geometry LR x0.1.

  --stage rel     : detached reliability calibrator.
                    Active: reliability_head only.
                    Loss: L_rel(s, y).
                    Frozen: everything else.

  --stage full    : original end-to-end behavior (back-compat / ablation).

Each stage logs an extended TensorBoard panel (planv2 section 6):
  loss/* trip/align/em/support/smooth/rel/cycle/sigma_reg/fold/photo/sup_corner
  H/offset_inf_px, H/offset_mean_px, H/kappa_log, H/fold_count
  q/mean, q/var, q/area_above_tau, q/A_v
  sigma/mean, residual/median
  eval_l2/direct, eval_l2/inverse, eval_l2/symmetric, eval_l2/identity, eval_l2/v1
  grad_norm/{joint_backbone, backbone, homography_head, posterior_head,
             uncertainty_head, reliability_head}
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
from losses.align import alignment_loss
from losses.em import em_posterior_loss
from losses.support import support_loss
from losses.smooth import edge_aware_smoothness
from losses.reliability import reliability_loss, build_invalid_pair_labels
from losses.cycle import cycle_loss
from losses.fold import fold_loss
from data.pairs import TrainPairDataset
from data.synth_pairs import SynthPairDataset
from data.invalid_pairs import build_invalid_batch
from data.test_dataset import TestDataset
from utils.eval_metrics import point_reprojection_error
from utils.inverse import safe_inverse_3x3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def patch_to_full_homography(H_patch: torch.Tensor, crop_xy: torch.Tensor) -> torch.Tensor:
    """Conjugate the patch-coord homography with the patch's translation to
    obtain the equivalent homography in full-image coordinates."""
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


def _freeze_for_stage(net, stage: str):
    """Configure which modules are trainable for a given stage."""
    all_mods = {
        "backbone":         net.backbone,
        "joint_backbone":   net.joint_backbone,
        "correlation":      net.correlation,
        "homography_head":  net.homography_head,
        "posterior_head":   net.posterior_head,
        "uncertainty_head": net.uncertainty_head,
        "reliability_head": net.reliability_head,
    }

    if stage == "synth":
        active = {"joint_backbone", "homography_head"}
    elif stage == "h_only":
        active = {"backbone", "joint_backbone", "correlation", "homography_head"}
    elif stage == "q_sigma":
        # planv2 Phase 5: train posterior and uncertainty WITHOUT corrupting H.
        # The H trunk (joint_backbone + correlation + homography_head) is
        # frozen so q/sigma learn against the converged alignment target
        # from h_only. Without this, the optimizer drifts H toward identity
        # on small-motion pairs while q/sigma chase the moving residual.
        active = {"backbone", "posterior_head", "uncertainty_head"}
    elif stage == "joint":
        active = {"backbone", "joint_backbone", "correlation", "homography_head",
                  "posterior_head", "uncertainty_head"}
    elif stage == "rel":
        active = {"reliability_head"}
    elif stage == "full":
        active = set(all_mods.keys())
    else:
        raise ValueError(f"unknown --stage: {stage}")

    for name, mod in all_mods.items():
        _set_requires_grad(mod, name in active)
    return active


def _make_param_groups(net, cfg: Config, stage: str):
    """Differential learning rates per module group. Only trainable params end
    up in the optimizer, so frozen modules contribute zero LR groups."""
    # Buckets
    h_mods = [net.joint_backbone, net.correlation, net.homography_head]
    bb_mods = [net.backbone]
    head_mods = [net.posterior_head, net.uncertainty_head, net.reliability_head]

    def _params(mods):
        ps = []
        for m in mods:
            ps += [p for p in m.parameters() if p.requires_grad]
        return ps

    groups = []
    p_h = _params(h_mods)
    p_b = _params(bb_mods)
    p_d = _params(head_mods)
    lr_h = cfg.lr_h
    lr_bb = cfg.lr_backbone
    lr_d = cfg.lr_heads

    # joint fine-tune lowers geometry LR by 10x (planv2 phase 6).
    if stage == "joint":
        lr_h *= 0.1
        lr_bb *= 0.1

    if p_h:
        groups.append({"params": p_h, "lr": lr_h})
    if p_b:
        groups.append({"params": p_b, "lr": lr_bb})
    if p_d:
        groups.append({"params": p_d, "lr": lr_d})
    if not groups:
        # Should not happen, but fallback to a single dummy group with
        # whatever is in the model to keep the optimizer constructible.
        groups.append({"params": list(net.parameters()), "lr": cfg.lr})
    return groups


def _save_checkpoint(net, save_dir: str, filename: str, **extra) -> str:
    """Save a checkpoint robustly. Recreates the directory if it's missing
    (defensive against external `rm`s or transient mount issues) and catches
    write errors -- disk-full or permission failures log a warning but do not
    crash training, so the next scheduled save can succeed once you free space.
    """
    path = os.path.join(save_dir, filename)
    try:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), **extra}, path)
        print(f"[ckpt] saved {path}", flush=True)
        return path
    except (OSError, RuntimeError) as e:
        print(f"[ckpt] FAILED to save {path}: {type(e).__name__}: {e}",
              flush=True)
        return ""


def _try_load_init(net, ckpt_path: str):
    """Soft-load a checkpoint (strict=False) when one is explicitly requested.

    Resolves the path against (in order): as-given (cwd-relative or absolute),
    then relative to the repo root. If a path was requested but neither
    resolution exists, RAISES rather than falling back to random init --
    silently training from scratch when staged init was intended would
    waste hours of compute. Pass --init_ckpt '' (default) to skip cleanly.
    """
    if not ckpt_path:
        return
    candidates = [ckpt_path]
    if not os.path.isabs(ckpt_path):
        candidates.append(os.path.normpath(os.path.join(_REPO_ROOT, ckpt_path)))
    resolved = next((c for c in candidates if os.path.isfile(c)), None)
    if resolved is None:
        raise FileNotFoundError(
            f"--init_ckpt was requested but the file was not found.\n"
            f"  Looked at: {candidates}\n"
            f"  CWD:       {os.getcwd()}\n"
            f"  Repo root: {_REPO_ROOT}\n"
            f"  Tip: pass an absolute path, or a path relative to the repo "
            f"root (e.g. 'train_log_v2/real_models/cdpc_synth_iter_*.pth'), "
            f"NOT relative to the Oneline-DLTv2/ directory."
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
EVAL_METRICS = ("direct", "inverse", "symmetric", "identity", "v1")


@torch.no_grad()
def run_eval_l2(net, test_loader, device, max_batches=None) -> dict:
    """Eval-L2 with all five conventions (planv2 section 2.5)."""
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

        err_ab  = point_reprojection_error(H_full,            pts_a, pts_b)
        err_ba  = point_reprojection_error(H_full,            pts_b, pts_a)
        err_inv = point_reprojection_error(H_full_inv,        pts_b, pts_a)
        err_id  = point_reprojection_error(_eye_like(H_full), pts_a, pts_b)

        per_pair = {
            "direct":    err_ab.mean(dim=1),
            "inverse":   err_inv.mean(dim=1),
            "symmetric": ((err_ab + err_inv) / 2.0).mean(dim=1),
            "identity":  err_id.mean(dim=1),
            "v1":        torch.minimum(err_ab, err_ba).mean(dim=1),
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
# Stage loops
# ---------------------------------------------------------------------------

def train(args, cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage = cfg.stage

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)

    # --- model ---
    net = CDPCNet(
        patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        backbone_pretrained=cfg.backbone_pretrained,
        corr_radius=cfg.corr_radius,
        corr_out_channels=cfg.corr_out_channels,
        bb_quarter_channels=cfg.bb_quarter_channels,
        bb_eighth_channels=cfg.bb_eighth_channels,
        homography_rho=cfg.homography_rho,
        post_init_prob=cfg.post_init_prob,
        sigma_min=cfg.sigma_min,
        use_normalized_dlt=cfg.use_normalized_dlt,
    ).to(device)
    # Pass init_ckpt through unmodified; _try_load_init does its own dual
    # resolution (cwd-relative, then repo-relative) and raises on miss.
    _try_load_init(net, cfg.init_ckpt)
    active = _freeze_for_stage(net, stage)
    print(f"[stage={stage}] active modules: {sorted(active)}", flush=True)
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    n_total = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"[stage={stage}] trainable: {n_train:.2f}M / {n_total:.2f}M", flush=True)

    # --- data ---
    if stage == "synth":
        train_ds = SynthPairDataset(
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
    optimizer = torch.optim.Adam(param_groups, amsgrad=True, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    # --- logging ---
    log_dir = _abs(cfg.log_dir)
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

            # --- invalid pairs only relevant in stages that train reliability
            do_rel = (stage in ("rel", "full")) and (glob_iter >= cfg.rel_warmup_iters)
            if do_rel:
                I_a_neg, I_b_neg, y_neg = build_invalid_batch(
                    I_a_patch, I_b_patch,
                    shuffle_frac=cfg.rel_shuffle_frac,
                    reshuffle_frac=cfg.rel_reshuffle_frac,
                )
                # For negatives, just duplicate the full images (they're
                # not what's compared anyway); the network learns from the
                # mismatched patches only.
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

            with torch.amp.autocast("cuda", enabled=False):
                F_b_nat        = out["F_b4"][:B].float()
                F_a_warped_nat = out["F_a_warped"][:B].float()
                F_a_nat        = out["F_a4"][:B].float()
                F_a_rec_nat    = out["F_a_recovered"][:B].float()
                q_nat          = out["q"][:B].float()
                log_sigma_nat  = out["log_sigma"][:B].float()
                residual_nat   = out["residual"][:B].float()
                valid_nat      = out["valid_mask"][:B].float()
                cycle_valid_nat = out["cycle_valid"][:B].float()
                cond_valid_nat  = out["cond_valid"][:B].float()
                offset_nat     = out["offset"][:B].float()
                s_all          = out["s"].float()

                # --- compute losses appropriate for the stage ---
                L = dict.fromkeys(
                    ["triplet", "align", "em", "support", "smooth", "rel",
                     "cycle", "sigma_reg", "fold", "photo", "sup_corner"],
                    torch.zeros((), device=device),
                )

                # L_fold is cheap and always on (where the H head trains).
                if stage in ("synth", "h_only", "q_sigma", "joint", "full"):
                    L["fold"] = fold_loss(offset_nat, cfg.patch_h, cfg.patch_w)

                if stage == "synth":
                    # Supervised corner loss.
                    offset_gt = batch["offset_gt"].to(device).float()
                    L["sup_corner"] = F.smooth_l1_loss(offset_nat, offset_gt, beta=1.0)
                else:
                    # Triplet: stage-aware q weighting.
                    use_q = (stage in ("q_sigma", "full")) and cfg.triplet_use_q_weighting
                    L["triplet"] = triplet_loss(
                        F_b_nat, F_a_warped_nat, F_a_nat,
                        q_nat, valid_nat, margin=cfg.triplet_margin,
                        use_q_weighting=use_q,
                    )

                # Photometric (Charbonnier on patch image-space residual).
                if stage in ("h_only", "q_sigma", "joint", "full"):
                    # Build a low-res patch residual on the feature-space residual map
                    # since we don't have an aligned image-space pair here. The
                    # `residual_nat` already is Charbonnier-suitable on |F_b - F_a_warped|.
                    photo = torch.sqrt(residual_nat ** 2 + 1e-6)
                    # valid-mask normalized mean
                    denom = valid_nat.sum(dim=(1, 2, 3)).clamp(min=1.0)
                    L["photo"] = ((photo * valid_nat).sum(dim=(1, 2, 3)) / denom).mean()

                # Alignment / EM / Support / Smooth (q + sigma stages).
                if stage in ("q_sigma", "joint", "full"):
                    if stage == "joint":
                        # q_dagger = sg(max(q, q_min_floor)); decouples H from
                        # transient q collapses (planv2 phase 6).
                        q_eff = torch.clamp(q_nat, min=cfg.q_min_floor).detach()
                    else:
                        q_eff = q_nat
                    L["align"] = alignment_loss(
                        residual_nat, log_sigma_nat, q_eff, valid_nat,
                    )

                    # Residual-gated L_em: only fire when residuals are
                    # credible (planv2 phase 5).
                    r_median = residual_nat.detach().median()
                    em_gated = (glob_iter >= cfg.em_warmup_iters and
                                float(r_median) <= cfg.em_residual_gate)
                    if em_gated:
                        L["em"] = em_posterior_loss(
                            q_nat, residual_nat, log_sigma_nat, valid_nat,
                            pi=cfg.em_prior_pi, r_max=cfg.em_r_max,
                        )
                    L["support"] = support_loss(q_nat, valid_nat, alpha=cfg.alpha_support)
                    L["smooth"] = edge_aware_smoothness(
                        q_nat, I_b_patch.float(), gamma=cfg.smoothness_gamma,
                    )
                    # Pull-to-1 sigma regularizer (kept simple in stages <=4;
                    # planv2's data-adaptive mu_sigma is a Phase-6 option not
                    # adopted here to avoid bad-H / large-sigma coupling).
                    L["sigma_reg"] = (log_sigma_nat * valid_nat).pow(2).sum() / \
                                     valid_nat.sum().clamp(min=1.0)

                if stage == "full":
                    L["cycle"] = cycle_loss(
                        F_a_nat, F_a_rec_nat, cycle_valid_nat, cond_valid_nat,
                    )

                if do_rel:
                    r_mean_nat = residual_nat.mean(dim=(1, 2, 3))
                    y_nat = build_invalid_pair_labels(
                        r_mean_nat,
                        hard_negative_percentile=cfg.rel_hard_neg_percentile,
                    )
                    y_target = torch.cat([y_nat, y_neg.float()], dim=0)
                    L["rel"] = reliability_loss(s_all, y_target)

                lam_align_eff = cfg.lambda_align * _ramp(
                    glob_iter, cfg.align_warmup_iters, cfg.align_ramp_iters)
                lam_em_eff = cfg.lambda_em * _ramp(
                    glob_iter, cfg.em_warmup_iters, cfg.em_ramp_iters)

                L_total = (
                    cfg.lambda_triplet   * L["triplet"]
                    + lam_align_eff      * L["align"]
                    + lam_em_eff         * L["em"]
                    + cfg.lambda_support * L["support"]
                    + cfg.lambda_smooth  * L["smooth"]
                    + cfg.lambda_rel     * L["rel"]
                    + cfg.lambda_cycle   * L["cycle"]
                    + cfg.lambda_sigma_reg * L["sigma_reg"]
                    + cfg.lambda_fold    * L["fold"]
                    + cfg.lambda_photo   * L["photo"]
                    + cfg.lambda_sup_corner * L["sup_corner"]
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
                kappa_H_nat = out["kappa_H"][:B].detach().float()
                kappa_log = float(torch.log(kappa_H_nat.clamp(min=1.0)).mean())
                # Fold-over count: any cross product flipped sign.
                # Counted here as the number of pairs where L["fold"] > 0.
                fold_count = float((L["fold"] > 0).float())
                # A_v: fraction of valid pixels.
                area_valid = float(valid_nat.mean())
                # q stats
                q_mean = float(q_nat.mean())
                q_var  = float(((q_nat - q_nat.mean()) ** 2).mean())
                q_area = float((q_nat > 0.5).float().mean())
                # sigma stats
                sigma_mean = float(log_sigma_nat.exp().mean())
                r_median   = float(residual_nat.median())

            for k, v in L.items():
                writer.add_scalar(f"loss/{k}", float(v), glob_iter)
            writer.add_scalar("loss/total", float(L_total), glob_iter)
            writer.add_scalar("opt/lr", optimizer.param_groups[0]["lr"], glob_iter)
            writer.add_scalar("opt/lambda_align_eff", lam_align_eff, glob_iter)
            writer.add_scalar("opt/lambda_em_eff",    lam_em_eff,    glob_iter)
            writer.add_scalar("H/offset_inf_px",  offset_inf,  glob_iter)
            writer.add_scalar("H/offset_mean_px", offset_mean, glob_iter)
            writer.add_scalar("H/kappa_log",      kappa_log,   glob_iter)
            writer.add_scalar("H/fold_count",     fold_count,  glob_iter)
            writer.add_scalar("q/mean",           q_mean,      glob_iter)
            writer.add_scalar("q/var",            q_var,       glob_iter)
            writer.add_scalar("q/area_above_tau", q_area,      glob_iter)
            writer.add_scalar("q/A_v",            area_valid,  glob_iter)
            writer.add_scalar("sigma/mean",       sigma_mean,  glob_iter)
            writer.add_scalar("residual/median",  r_median,    glob_iter)

            # Per-module grad norms.
            if glob_iter % 100 == 0:
                for mod_name, mod in [
                    ("backbone",         net.backbone),
                    ("joint_backbone",   net.joint_backbone),
                    ("homography_head",  net.homography_head),
                    ("posterior_head",   net.posterior_head),
                    ("uncertainty_head", net.uncertainty_head),
                    ("reliability_head", net.reliability_head),
                ]:
                    g2 = 0.0
                    n = 0
                    for p in mod.parameters():
                        if p.grad is not None:
                            g2 += float(p.grad.norm() ** 2)
                            n += 1
                    if n > 0:
                        writer.add_scalar(f"grad_norm/{mod_name}", g2 ** 0.5, glob_iter)

            if batch_idx % cfg.score_print_freq == 0:
                msg = (f"[ep {epoch+1:02d} it {glob_iter:06d} stage={stage}] "
                       f"L={float(L_total):.4f}  "
                       f"sup={float(L['sup_corner']):.3f} "
                       f"trip={float(L['triplet']):.3f} "
                       f"al={float(L['align']):.3f} "
                       f"em={float(L['em']):.3f} "
                       f"sup={float(L['support']):.3f} "
                       f"sm={float(L['smooth']):.4f} "
                       f"rel={float(L['rel']):.3f} "
                       f"cyc={float(L['cycle']):.3f} "
                       f"ph={float(L['photo']):.3f} "
                       f"fold={float(L['fold']):.3f}  "
                       f"q={q_mean:.3f} sig={sigma_mean:.3f} "
                       f"r_med={r_median:.3f} "
                       f"off_inf={offset_inf:.2f}px "
                       f"off_avg={offset_mean:.2f}px "
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

            if glob_iter > 0 and glob_iter % cfg.model_save_freq == 0:
                _save_checkpoint(
                    net, save_dir,
                    f"cdpc_{stage}_iter_{glob_iter}.pth",
                    iter=glob_iter, epoch=epoch, stage=stage,
                    config=cfg.__dict__,
                )

            glob_iter += 1

            # synth has a per-epoch iteration cap, since the dataset is
            # generative and we want fast supervised bootstrap not a full
            # pass through every line of Train_List.
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
    else:
        print(f"[done] training finished but FINAL checkpoint failed to save; "
              f"check disk/permissions on {save_dir}", flush=True)
    writer.close()


def _parse_args(cfg: Config):
    p = argparse.ArgumentParser()
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--max_epoch", type=int, default=cfg.max_epoch)
    p.add_argument("--lr",       type=float, default=cfg.lr)
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
    p.add_argument("--stage",     type=str, default=cfg.stage,
                   choices=["synth", "h_only", "q_sigma", "joint", "rel", "full"])
    p.add_argument("--init_ckpt", type=str, default=cfg.init_ckpt)
    p.add_argument("--max_frame_gap", type=int, default=0,
                   help="Stage 3 curriculum: max abs(frame_b - frame_a). 0 = no filter.")
    p.add_argument("--homography_rho", type=float, default=cfg.homography_rho,
                   help="Bound for H head: |corner_offset| <= rho. Must be >= synth_rho_max.")
    p.add_argument("--synth_rho_max", type=int, default=cfg.synth_rho_max,
                   help="Per-corner perturbation range for the synth dataset, in px.")
    p.add_argument("--model_save_freq", type=int, default=cfg.model_save_freq,
                   help="Iters between checkpoint saves. Lower it to keep more "
                        "intermediate checkpoints if disk space allows.")
    p.add_argument("--lambda_triplet", type=float, default=cfg.lambda_triplet,
                   help="Weight on L_triplet. Lower for small-motion h_only "
                        "where triplet stays pinned at the margin.")
    p.add_argument("--lambda_photo", type=float, default=cfg.lambda_photo,
                   help="Weight on L_photo (Charbonnier on feature residual). "
                        "Raise for h_only so the alignment signal isn't drowned "
                        "by a stuck triplet.")
    p.add_argument("--triplet_margin", type=float, default=cfg.triplet_margin,
                   help="Triplet hinge margin. Default 1.0 is large vs feature "
                        "residual scale (~0.05); lower to 0.05-0.1 if L_trip "
                        "sticks near margin for many iters.")
    args = p.parse_args()

    # Push CLI args back into cfg so downstream code reads a single source.
    cfg.batch_size = args.batch_size
    cfg.max_epoch = args.max_epoch
    cfg.lr = args.lr
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
    cfg.homography_rho = args.homography_rho
    cfg.synth_rho_max = args.synth_rho_max
    cfg.model_save_freq = args.model_save_freq
    cfg.lambda_triplet = args.lambda_triplet
    cfg.lambda_photo = args.lambda_photo
    cfg.triplet_margin = args.triplet_margin
    if cfg.homography_rho < cfg.synth_rho_max:
        raise ValueError(
            f"homography_rho ({cfg.homography_rho}) must be >= synth_rho_max "
            f"({cfg.synth_rho_max}); otherwise the synth target is unreachable "
            f"and L_sup_corner plateaus while offset_inf_px saturates at rho."
        )
    return args


if __name__ == "__main__":
    cfg = Config()
    args = _parse_args(cfg)
    print(f"[config] {cfg}", flush=True)
    train(args, cfg)
