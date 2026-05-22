"""Single source of truth for hyperparameters. Edit here or override via CLI
in train.py / eval.py.

planv3 update: new architecture is a Siamese feature pyramid (1/4, 1/8, 1/16)
followed by a 3-level coarse-to-fine homography regressor with residual
DLT updates. The joint 2-channel backbone is removed; the per-image trunk
is the only feature path. Stage curriculum splits the old `q_sigma` into
`q_only` and `sigma_only` to decouple the two heads' training.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Config:
    # ---- Data
    img_h: int = 360
    img_w: int = 640
    patch_h: int = 315
    patch_w: int = 560
    rho: int = 16

    # ---- Backbone / feature pyramid (planv3 §3.2)
    backbone_pretrained: bool = True
    bb_quarter_channels: int = 64       # 1/4 scale
    bb_eighth_channels:  int = 128      # 1/8 scale
    bb_sixteenth_channels: int = 256    # 1/16 scale (NEW for v3)

    # ---- Local correlation
    corr_radius: int = 4                # ±4 cells at each pyramid level
    corr_out_channels: int = 32

    # ---- Homography pyramid (planv3 §3.3)
    # Three coarse-to-fine levels at strides 1/16, 1/8, 1/4. Per-level corner
    # offset bound rho_t in input pixels. Sum = 32+16+8 = 56 px capture basin.
    homography_levels: int = 3
    rho_per_level: Tuple[float, float, float] = (32.0, 16.0, 8.0)
    # Single-level legacy bound, kept so back-compat checkpoints can load.
    homography_rho: float = 32.0

    # ---- Pixel heads (planv3 §3.4)
    # Posterior bias init so q ≈ post_init_prob at iter 0.
    post_init_prob: float = 0.7
    # Stage-dependent sigma floor. The CLI/train code overrides this per stage
    # (high during q/sigma training to dodge the Kendall-Gal trap, low later
    # once H and q are stable).
    sigma_min: float = 0.5

    # ---- EM posterior (planv3 §4.3)
    em_prior_pi: float = 0.5
    em_r_max: float = 4.0
    em_warmup_iters: int = 5000
    # Replaced by self-paced exp gate when use_em_self_paced_gate=True.
    em_residual_gate: float = 1.5
    use_em_self_paced_gate: bool = True
    em_self_paced_beta: float = 2.0   # exp(-beta * med(r) / r_max)

    # ---- Alignment loss split (planv3 §4.2)
    # `align_soft` drives H/q/trunk (no sigma, fixed denom).
    # `align_het`  drives sigma only (Kendall-Gal on q-selected pixels with sg(sigma)).
    align_warmup_iters: int = 3000
    align_ramp_iters: int = 2000
    em_ramp_iters: int = 2000
    q_select_tau: float = 0.5             # hard q threshold for L_align_het selection set

    # ---- Reliability head
    rel_shuffle_frac: float = 0.5
    rel_reshuffle_frac: float = 0.25
    rel_hard_neg_percentile: float = 0.85
    rel_warmup_iters: int = 7000

    # ---- Cycle loss
    cycle_cond_max: float = 1.0e4

    # ---- q-dagger (joint stage)
    q_min_floor: float = 0.25          # q_dagger = sg(max(q, q_min_floor))

    # ---- Loss weights (planv3 §5)
    lambda_triplet:    float = 0.5
    lambda_align_soft: float = 0.5     # ramps to 1.0
    lambda_align_het:  float = 0.3
    lambda_em:         float = 0.1     # ramps to 0.2
    lambda_support:    float = 0.1
    lambda_smooth:     float = 1.0e-3
    lambda_rel:        float = 1.0
    lambda_cycle:      float = 0.05
    lambda_sigma:      float = 0.1
    lambda_fold:       float = 0.1
    lambda_photo_img:  float = 0.5     # planv3 B5: image-space, not feature
    lambda_sup_corner: float = 1.0
    lambda_sup_H:      float = 0.1     # Frobenius supervision (synth stage)

    alpha_support: float = 0.5         # planv3 B4 raised from 0.25
    triplet_use_q_weighting: bool = False
    smoothness_gamma: float = 10.0
    triplet_margin: float = 1.0

    # ---- Legacy loss weights (kept so old configs load; ignored by v3 loop)
    lambda_align: float = 1.0          # superseded by align_soft + align_het
    lambda_sigma_reg: float = 0.1      # superseded by lambda_sigma (log-r prior)
    lambda_photo: float = 0.0          # superseded by lambda_photo_img

    # ---- Optimizer (differential LR per parameter group)
    lr: float = 1.0e-4                 # default LR for all groups when --stage full
    lr_h: float = 2.0e-4               # H trunk + head LR for staged runs
    lr_backbone: float = 2.0e-5        # per-image backbone (slower)
    lr_heads: float = 1.0e-4           # q, sigma, reliability heads
    weight_decay: float = 1.0e-4
    lr_gamma: float = 0.8              # legacy ExponentialLR fallback (used iff use_cosine_lr=False)
    use_cosine_lr: bool = True         # planv3 §7
    batch_size: int = 16
    max_epoch: int = 30
    grad_clip: float = 1.0

    # ---- Multi-stage training (planv3 §6)
    # New stage list:
    #   synth     : supervised geometric bootstrap (3-level corner Huber + L_sup_H + fold)
    #   h_only    : unsupervised H refinement (image photometric + triplet + fold)
    #   q_only    : posterior head only (EM + support + smooth), H/sigma frozen
    #   sigma_only: uncertainty head only (Kendall-Gal het + sigma prior), H/q frozen
    #   joint     : everything with q_dagger and detached sigma->released
    #   rel       : reliability calibrator on detached phi
    #   full      : legacy end-to-end ablation (back-compat with v2 runs)
    stage: str = "full"
    init_ckpt: str = ""                # checkpoint to resume from (any stage)

    # ---- Synthetic-H (Stage 1)
    synth_rho_max: int = 32
    synth_iters_per_epoch: int = 4000
    use_v3_synth: bool = True          # planv3 B6: structured-affine augmentation

    # ---- DLT
    use_normalized_dlt: bool = True

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
    log_dir: str = "train_log_v3/logs"
    model_save_dir: str = "train_log_v3/real_models"

    # ---- Legacy fields kept for back-compat with older checkpoints/configs.
    log_sigma_min: float = -3.0
    log_sigma_max: float = 5.0
