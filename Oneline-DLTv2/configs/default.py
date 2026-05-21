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
    homography_rho: float = 16.0       # was 32; smaller rho + larger FC init
                                        # together give initial offsets in the
                                        # 5-10 px range, enough for the triplet
                                        # to drive H out of the identity basin.
    post_init_prob: float = 0.7
    log_sigma_min: float = -5.0
    log_sigma_max: float = 5.0

    # ---- EM posterior
    em_prior_pi: float = 0.5
    em_r_max: float = 4.0
    em_warmup_iters: int = 5000     # disable L_em during this many iters

    # ---- Alignment ramp
    # L_align is disabled (lambda=0) for `align_warmup_iters`, then linearly
    # ramped to `lambda_align` over `align_ramp_iters`. Reason: at init H is
    # near identity so triplet has weak gradient; if L_align fires immediately
    # the network exploits sigma-shrinking instead of learning H.
    align_warmup_iters: int = 3000
    align_ramp_iters: int = 2000
    em_ramp_iters: int = 2000

    # ---- Reliability head
    rel_shuffle_frac: float = 0.5
    rel_reshuffle_frac: float = 0.25
    rel_hard_neg_percentile: float = 0.85
    rel_warmup_iters: int = 7000    # disable L_rel during this many iters

    # ---- Cycle loss
    cycle_cond_max: float = 1.0e4

    # ---- Loss weights
    lambda_triplet: float = 1.0
    lambda_align: float = 1.0
    lambda_em: float = 0.1             # was 0.5; EM raw values 3-5 contributed
                                        # ~2.5 to L_total at warmup end, causing
                                        # an optimizer shock that distorted H.
    lambda_support: float = 0.1        # was 0.01; needs to actually bite to
                                        # prevent q-collapse below alpha.
    lambda_smooth: float = 1.0e-3
    lambda_rel: float = 0.1
    lambda_cycle: float = 0.0          # off by default: identity is the unique
                                        # global minimizer of the analytic cycle,
                                        # so this loss actively pulls H toward
                                        # identity during early training. Turn
                                        # on (e.g. 0.01) only for the ablation
                                        # row in the paper.
    lambda_sigma_reg: float = 0.01     # soft pull of log(sigma) toward 0
                                        # (sigma toward 1); blocks the Kendall-
                                        # Gal sigma-shrinking shortcut without
                                        # hard-clamping log_sigma.
    alpha_support: float = 0.25        # was 0.10; observed q_bar collapsing
                                        # to 0.04 during align ramp, way below
                                        # the floor.
    triplet_use_q_weighting: bool = False  # plan-literal triplet had q weight,
                                        # but that creates a degenerate loop:
                                        # q collapse -> tiny triplet grad ->
                                        # H stops learning -> worse q. Uniform
                                        # weighting decouples H learning from q.
    smoothness_gamma: float = 10.0
    triplet_margin: float = 1.0

    # ---- Optimizer
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    lr_gamma: float = 0.8           # exponential decay per epoch
    batch_size: int = 16
    max_epoch: int = 30
    grad_clip: float = 1.0

    # ---- Logging
    score_print_freq: int = 200
    viz_freq: int = 500
    model_save_freq: int = 4000
    eval_every: int = 2000          # run eval-L2 every N iterations

    # ---- AMP
    use_amp: bool = True

    # ---- Paths (relative to repo root unless absolute)
    train_list: str = "Data/Train_List.txt"
    train_root: str = "Data/Train"
    test_root: str = "."             # test_dataset.py joins this with 'Data/...'
    log_dir: str = "train_log_v2/logs"
    model_save_dir: str = "train_log_v2/real_models"
