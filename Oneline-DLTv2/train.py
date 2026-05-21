"""Train CDPC v2 with periodic eval-L2 monitoring (the missing-feature
complaint from the earlier v2 attempt is fixed here)."""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
def _make_summary_writer_cls():
    """Resolve a SummaryWriter implementation. Order:
        1. torch.utils.tensorboard           (preferred; bundled with torch)
        2. tensorboardX                       (fallback; no TF dependency)
        3. _NoOpWriter                        (stdout only; non-fatal)
    The TF-bundled tensorboard build can fail to import on some Python
    environments (bfloat16 / numpy ABI mismatch); we handle that gracefully.
    """
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
# Put the package dir first so absolute imports of model/utils/losses/data work
# even though the folder name contains a hyphen.
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
from data.pairs import TrainPairDataset
from data.invalid_pairs import build_invalid_batch
from data.test_dataset import TestDataset
from utils.eval_metrics import point_reprojection_error
from utils.viz import log_training_panel


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


@torch.no_grad()
def run_eval_l2(net, test_loader, device, max_batches=None) -> dict:
    """Returns mean L2 reprojection error overall and per scene category."""
    net.eval()
    per_scene = {}
    all_errs = []
    for i, batch in enumerate(test_loader):
        if max_batches is not None and i >= max_batches:
            break
        I_a = batch["I_a_patch"].to(device, non_blocking=True)
        I_b = batch["I_b_patch"].to(device, non_blocking=True)
        crop_xy = batch["crop_xy"].to(device, non_blocking=True)

        out = net(I_a, I_b)
        H_full = patch_to_full_homography(out["H_ab"], crop_xy)

        pts = batch["points"].to(device)               # (B, K, 2, 2): K pairs of (a, b)
        pts_a = pts[:, :, 0, :]
        pts_b = pts[:, :, 1, :]
        err_ab = point_reprojection_error(H_full, pts_a, pts_b)
        # Symmetry (v1 takes min(err_LR, err_RL)).
        err_ba = point_reprojection_error(H_full, pts_b, pts_a)
        err = torch.minimum(err_ab.mean(dim=1), err_ba.mean(dim=1))   # (B,)
        for j in range(I_a.shape[0]):
            scene = batch["scene"][j]
            per_scene.setdefault(scene, []).append(float(err[j]))
            all_errs.append(float(err[j]))
    net.train()
    summary = {"overall": float(np.mean(all_errs)) if all_errs else float("nan")}
    for k, v in per_scene.items():
        summary[k] = float(np.mean(v)) if v else float("nan")
    return summary


def train(args, cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)

    train_ds = TrainPairDataset(
        _abs(cfg.train_list), _abs(cfg.train_root),
        patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        img_h=cfg.img_h, img_w=cfg.img_w, rho=cfg.rho,
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

    print(f"[train] {len(train_ds)} pairs, {len(test_ds)} test pairs", flush=True)

    net = CDPCNet(
        patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        backbone_pretrained=cfg.backbone_pretrained,
        corr_radius=cfg.corr_radius,
        corr_out_channels=cfg.corr_out_channels,
        bb_quarter_channels=cfg.bb_quarter_channels,
        bb_eighth_channels=cfg.bb_eighth_channels,
        homography_rho=cfg.homography_rho,
        post_init_prob=cfg.post_init_prob,
        log_sigma_min=cfg.log_sigma_min,
        log_sigma_max=cfg.log_sigma_max,
    ).to(device)
    print(f"[train] model: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params",
          flush=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                                 amsgrad=True, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

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
        for batch_idx, (I_a, I_b, _crop_xy) in enumerate(train_loader):
            I_a = I_a.to(device, non_blocking=True)
            I_b = I_b.to(device, non_blocking=True)
            B = I_a.shape[0]

            do_rel = glob_iter >= cfg.rel_warmup_iters
            if do_rel:
                I_a_neg, I_b_neg, y_neg = build_invalid_batch(
                    I_a, I_b,
                    shuffle_frac=cfg.rel_shuffle_frac,
                    reshuffle_frac=cfg.rel_reshuffle_frac,
                )
                I_a_all = torch.cat([I_a, I_a_neg], dim=0)
                I_b_all = torch.cat([I_b, I_b_neg], dim=0)
            else:
                I_a_all, I_b_all = I_a, I_b
                y_neg = None

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                out = net(I_a_all, I_b_all)

                # Slice natural-pair outputs.
                F_b_nat        = out["F_b4"][:B]
                F_a_warped_nat = out["F_a_warped"][:B]
                F_a_nat        = out["F_a4"][:B]
                F_a_rec_nat    = out["F_a_recovered"][:B]
                q_nat          = out["q"][:B]
                log_sigma_nat  = out["log_sigma"][:B]
                residual_nat   = out["residual"][:B]
                valid_nat      = out["valid_mask"][:B]
                cycle_valid_nat = out["cycle_valid"][:B]
                cond_valid_nat  = out["cond_valid"][:B]
                s_all          = out["s"]

                L_triplet = triplet_loss(
                    F_b_nat, F_a_warped_nat, F_a_nat,
                    q_nat, valid_nat, margin=cfg.triplet_margin,
                )
                L_align = alignment_loss(
                    residual_nat, log_sigma_nat, q_nat, valid_nat,
                )
                if glob_iter >= cfg.em_warmup_iters:
                    L_em = em_posterior_loss(
                        q_nat, residual_nat, log_sigma_nat, valid_nat,
                        pi=cfg.em_prior_pi, r_max=cfg.em_r_max,
                    )
                else:
                    L_em = torch.zeros((), device=device)

                L_support = support_loss(q_nat, valid_nat, alpha=cfg.alpha_support)
                L_smooth = edge_aware_smoothness(q_nat, I_b, gamma=cfg.smoothness_gamma)

                if do_rel:
                    r_mean_nat = residual_nat.mean(dim=(1, 2, 3))
                    y_nat = build_invalid_pair_labels(
                        r_mean_nat,
                        hard_negative_percentile=cfg.rel_hard_neg_percentile,
                    )
                    y_target = torch.cat([y_nat, y_neg], dim=0)
                    L_rel = reliability_loss(s_all, y_target)
                else:
                    L_rel = torch.zeros((), device=device)

                L_cycle = cycle_loss(
                    F_a_nat, F_a_rec_nat, cycle_valid_nat, cond_valid_nat,
                )

                L_total = (
                    cfg.lambda_triplet * L_triplet
                    + cfg.lambda_align   * L_align
                    + cfg.lambda_em      * L_em
                    + cfg.lambda_support * L_support
                    + cfg.lambda_smooth  * L_smooth
                    + cfg.lambda_rel     * L_rel
                    + cfg.lambda_cycle   * L_cycle
                )

            scaler.scale(L_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("loss/total",   float(L_total),   glob_iter)
            writer.add_scalar("loss/triplet", float(L_triplet), glob_iter)
            writer.add_scalar("loss/align",   float(L_align),   glob_iter)
            writer.add_scalar("loss/em",      float(L_em),      glob_iter)
            writer.add_scalar("loss/support", float(L_support), glob_iter)
            writer.add_scalar("loss/smooth",  float(L_smooth),  glob_iter)
            writer.add_scalar("loss/rel",     float(L_rel),     glob_iter)
            writer.add_scalar("loss/cycle",   float(L_cycle),   glob_iter)
            writer.add_scalar("opt/lr",       scheduler.get_last_lr()[0], glob_iter)
            writer.add_scalar("q/mean",       float(q_nat.mean()), glob_iter)
            writer.add_scalar("sigma/mean",   float(log_sigma_nat.exp().mean()), glob_iter)

            if batch_idx % cfg.score_print_freq == 0:
                msg = (f"[ep {epoch+1:02d} it {glob_iter:06d}] "
                       f"L={float(L_total):.4f}  "
                       f"trip={float(L_triplet):.3f} align={float(L_align):.3f} "
                       f"em={float(L_em):.3f} sup={float(L_support):.3f} "
                       f"sm={float(L_smooth):.4f} rel={float(L_rel):.3f} "
                       f"cyc={float(L_cycle):.3f} q_bar={float(q_nat.mean()):.3f}")
                print(msg, flush=True)

            if glob_iter % cfg.viz_freq == 0:
                Hh, Ww = I_a.shape[-2:]
                pred_I_b = F.interpolate(F_a_warped_nat[:1, :1].detach(),
                                         size=(Hh, Ww), mode="bilinear",
                                         align_corners=True)
                q_viz  = F.interpolate(q_nat[:1].detach(), size=(Hh, Ww),
                                       mode="bilinear", align_corners=True)
                ls_viz = F.interpolate(log_sigma_nat[:1].detach(), size=(Hh, Ww),
                                       mode="bilinear", align_corners=True)
                r_viz  = F.interpolate(residual_nat[:1].detach(), size=(Hh, Ww),
                                       mode="bilinear", align_corners=True)
                log_training_panel(
                    writer, glob_iter,
                    I_a=I_a[:1], I_b=I_b[:1], pred_I_b=pred_I_b,
                    q_map=q_viz, log_sigma_map=ls_viz, residual_map=r_viz,
                )

            # ---- eval-L2 monitoring (matches v1 expectation) ----
            if glob_iter > 0 and glob_iter % cfg.eval_every == 0:
                t0 = time.time()
                summary = run_eval_l2(net, test_loader, device,
                                      max_batches=args.eval_max_batches)
                dt = time.time() - t0
                for k, v in summary.items():
                    writer.add_scalar(f"eval_l2/{k}", v, glob_iter)
                print(f"[eval @ it {glob_iter}] " +
                      "  ".join(f"{k}={v:.3f}" for k, v in summary.items()) +
                      f"  ({dt:.1f}s)", flush=True)

            if glob_iter > 0 and glob_iter % cfg.model_save_freq == 0:
                ckpt = os.path.join(save_dir, f"cdpc_iter_{glob_iter}.pth")
                torch.save({"state_dict": net.state_dict(),
                            "iter": glob_iter, "epoch": epoch,
                            "config": cfg.__dict__}, ckpt)
                print(f"[ckpt] saved {ckpt}", flush=True)

            glob_iter += 1

        scheduler.step()

    final_ckpt = os.path.join(save_dir, f"cdpc_iter_{glob_iter}_final.pth")
    torch.save({"state_dict": net.state_dict(),
                "iter": glob_iter, "epoch": cfg.max_epoch,
                "config": cfg.__dict__}, final_ckpt)
    print(f"[done] final checkpoint: {final_ckpt}", flush=True)
    writer.close()


def _parse_args(cfg: Config):
    p = argparse.ArgumentParser()
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--max_epoch", type=int, default=cfg.max_epoch)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--patch_h", type=int, default=cfg.patch_h)
    p.add_argument("--patch_w", type=int, default=cfg.patch_w)
    p.add_argument("--img_h", type=int, default=cfg.img_h)
    p.add_argument("--img_w", type=int, default=cfg.img_w)
    p.add_argument("--amp", action="store_true", default=cfg.use_amp)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--eval_every", type=int, default=cfg.eval_every)
    p.add_argument("--eval_max_batches", type=int, default=None,
                   help="If set, evaluate on at most this many test batches.")
    p.add_argument("--train_list", type=str, default=cfg.train_list)
    p.add_argument("--train_root", type=str, default=cfg.train_root)
    p.add_argument("--log_dir", type=str, default=cfg.log_dir)
    p.add_argument("--model_save_dir", type=str, default=cfg.model_save_dir)
    args = p.parse_args()
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
    return args


if __name__ == "__main__":
    cfg = Config()
    args = _parse_args(cfg)
    print(f"[config] {cfg}", flush=True)
    train(args, cfg)
