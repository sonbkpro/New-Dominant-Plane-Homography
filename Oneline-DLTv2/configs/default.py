"""Single source of truth for hyperparameters. Edit here or override via CLI
in train.py / eval.py."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Config:
    # ---- Data
    img_h: int = 360
    img_w: int = 640
    patch_h: int = 315
    patch_w: int = 560
    rho: int = 16

    # ---- Backbone / architecture
    backbone_pretrained: bool = True
    bb_quarter_channels: int = 64
    bb_eighth_channels: int = 128
    corr_radius: int = 4
    corr_out_channels: int = 32

    # Bounded H output via rho_max * tanh. The network can never produce a
    # corner offset whose magnitude exceeds this, which kills the runaway-H
    # failure mode and lets us train with a higher LR. MUST be >= synth_rho_max
    # below, else the synth-stage supervised target is unreachable and L_sup
    # plateaus while `H/offset_inf_px` saturates at this value.
    homography_rho: float = 32.0

    # Posterior bias init so q ~= post_init_prob at iter 0.
    post_init_prob: float = 0.7

    # Uncertainty parameterized as sigma = sigma_min + softplus(u). u=0 gives
    # sigma = sigma_min + log(2) ~= 0.74 at init.
    sigma_min: float = 0.05

    # ---- EM posterior
    em_prior_pi: float = 0.5
    em_r_max: float = 4.0
    em_warmup_iters: int = 5000
    # Once we leave warmup, L_em fires only if the current batch's residual
    # median is below this threshold. Stops EM from following bad-H residuals.
    em_residual_gate: float = 1.5

    # ---- Alignment ramp
    align_warmup_iters: int = 3000
    align_ramp_iters: int = 2000
    em_ramp_iters: int = 2000

    # ---- Reliability head
    rel_shuffle_frac: float = 0.5
    rel_reshuffle_frac: float = 0.25
    rel_hard_neg_percentile: float = 0.85
    rel_warmup_iters: int = 7000

    # ---- Cycle loss
    cycle_cond_max: float = 1.0e4

    # ---- q-dagger (Stage 5 joint fine-tune)
    q_min_floor: float = 0.25     # q_dagger = sg(max(q, q_min_floor))

    # ---- Loss weights
    lambda_triplet: float = 1.0
    lambda_align: float = 1.0
    lambda_em: float = 0.1
    lambda_support: float = 0.1
    lambda_smooth: float = 1.0e-3
    lambda_rel: float = 0.1
    lambda_cycle: float = 0.0
    lambda_sigma_reg: float = 0.1
    lambda_fold: float = 0.1          # NEW: fold-over penalty on quad
    lambda_photo: float = 0.25        # NEW: photometric (Stage 3) anchor
    lambda_sup_corner: float = 1.0    # NEW: supervised corner L1 (Stage 2)

    alpha_support: float = 0.25
    triplet_use_q_weighting: bool = False
    smoothness_gamma: float = 10.0
    triplet_margin: float = 1.0

    # ---- Optimizer (differential LR per parameter group)
    lr: float = 1.0e-4               # default LR for all groups when --stage full
    lr_h: float = 2.0e-4             # H trunk + head LR for staged runs
    lr_backbone: float = 2.0e-5      # per-image backbone (slower)
    lr_heads: float = 1.0e-4         # q, sigma, reliability heads
    weight_decay: float = 1.0e-4
    lr_gamma: float = 0.8
    batch_size: int = 16
    max_epoch: int = 30
    grad_clip: float = 1.0

    # ---- Multi-stage training
    # Stages (planv2):
    #   "synth"  : synthetic supervised H bootstrap. joint_backbone+H head only.
    #   "h_only" : real-pair H-only. + per-image backbone. q/sigma/rel frozen.
    #   "q_sigma": adds posterior + uncertainty. rel frozen.
    #   "joint"  : joint fine-tune with q_dagger weighting. rel frozen.
    #   "rel"    : detached reliability calibrator only.
    #   "full"   : original end-to-end behavior (back-compat / ablation).
    stage: str = "full"
    init_ckpt: str = ""              # checkpoint to resume from (any stage)

    # ---- Synthetic-H (Stage 2)
    synth_rho_max: int = 32           # max per-corner perturbation (px)
    synth_iters_per_epoch: int = 4000 # iter cap when --stage synth

    # ---- DLT
    use_normalized_dlt: bool = True   # Hartley-normalized DLT

    # ---- Logging
    score_print_freq: int = 200
    viz_freq: int = 500
    model_save_freq: int = 4000
    eval_every: int = 2000

    # ---- AMP
    use_amp: bool = True

    # ---- Paths (relative to repo root unless absolute)
    train_list: str = "Data/Train_List.txt"
    train_root: str = "Data/Train"
    test_root: str = "."
    log_dir: str = "train_log_v2/logs"
    model_save_dir: str = "train_log_v2/real_models"

    # ---- Legacy fields kept for back-compat with older checkpoints/configs.
    # New code reads sigma_min instead; these are no-ops in the new pipeline.
    log_sigma_min: float = -3.0
    log_sigma_max: float = 5.0
