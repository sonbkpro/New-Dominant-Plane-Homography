"""Single source of truth for hyperparameters. Edit here or override via CLI
in train.py / eval.py.

CDPC-v4 redesign:
  - Joint 2-channel ResNet-34 geometry trunk (v1-style) replaces the Siamese
    feature pyramid as the H regressor.
  - Sub-pixel correlation refiner at 1/4 provides residual ΔH composed on
    H_init for sub-pixel accuracy.
  - q, σ, s heads run downstream on DETACHED features -- they cannot collapse
    the geometry features, so the entire planv3 Fix-A/B/C postmortem failure
    mode is structurally eliminated.
  - 4 stages: synth, geom, cdpc, rel (plus `full` for end-to-end ablation).
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

    # ---- Geometry trunk (CDPC-v4)
    trunk_pretrained: bool = False           # v1 trained from random init; pretrained ImageNet weights typically hurt
    trunk_rho_init: float = 32.0             # bound on H_init corner offsets (px); pyramid budget was 56 -- 32 is enough for ±32px synth + headroom
    share_feature_channels: int = 1          # ShareFeature output channels (1 = v1 default)

    # ---- Sub-pixel correlation refiner (CDPC-v4)
    refiner_feat_channels: int = 64          # per-image feature channels at 1/4
    refiner_radius: int = 6                  # ± half-extent of the local cost volume (feature pixels)
    refiner_pretrained: bool = False
    refiner_corner_window: int = 7           # corner aggregation window in *feature pixels*
    refiner_temperature: float = 1.0         # cost-volume softmax temperature; <1 sharpens, >1 smooths
    refiner_rho_max: float = 16.0            # bound on residual corner-offset (px)

    # ---- DLT
    use_normalized_dlt: bool = True

    # ---- Pixel heads
    post_init_prob: float = 0.7              # PosteriorHead bias init so q ≈ 0.7 at iter 0
    sigma_min: float = 0.5                   # UncertaintyHead floor on sigma

    # ---- EM posterior
    em_prior_pi: float = 0.5
    em_r_max: float = 4.0
    em_warmup_iters: int = 2000              # cdpc stage trains q from iter 0; warmup only matters in `full`
    em_residual_gate: float = 1.5            # legacy hard gate; unused when self-paced is on
    use_em_self_paced_gate: bool = True
    em_self_paced_beta: float = 2.0

    # ---- Alignment-het loss (sigma supervision via Kendall-Gål on q-selected set)
    align_warmup_iters: int = 2000           # used only in `full` stage
    align_ramp_iters: int = 2000
    em_ramp_iters: int = 2000
    q_select_tau: float = 0.5

    # ---- Reliability head
    rel_shuffle_frac: float = 0.5
    rel_reshuffle_frac: float = 0.25
    rel_hard_neg_percentile: float = 0.85
    rel_warmup_iters: int = 2000             # `rel` stage trains from 0; only matters in `full`

    # ---- Cycle loss
    cycle_cond_max: float = 1.0e4

    # ---- Loss weights
    lambda_triplet:    float = 1.0           # CDPC-v4 makes triplet the primary geometry signal in `geom`
    lambda_align_het:  float = 0.3           # Kendall-Gål heteroscedastic on q-selected set (cdpc stage)
    lambda_em:         float = 0.5           # ramps up after em_warmup_iters
    lambda_support:    float = 0.1
    lambda_smooth:     float = 1.0e-3
    lambda_rel:        float = 1.0
    lambda_cycle:      float = 0.05
    lambda_sigma:      float = 0.1           # log-residual prior on log_sigma
    lambda_fold:       float = 0.1
    lambda_photo_img:  float = 1.0           # image-space Charbonnier; primary H anchor alongside triplet
    lambda_sup_corner: float = 1.0           # supervised Huber on corner offsets (synth stage)
    lambda_sup_H:      float = 0.1           # Frobenius supervision on full H (synth stage)

    alpha_support: float = 0.5
    triplet_margin: float = 1.0
    triplet_use_q_weighting: bool = False
    smoothness_gamma: float = 10.0

    # ---- Optimizer (differential LR per parameter group)
    lr: float = 1.0e-4                       # fallback when --stage full
    lr_trunk:    float = 1.0e-4              # joint geometry trunk (ResNet-34)
    lr_refiner:  float = 1.0e-4              # sub-pixel correlation refiner
    lr_heads:    float = 1.0e-4              # CDPC heads (q, sigma, reliability)
    weight_decay: float = 1.0e-4
    lr_gamma: float = 0.8                    # ExponentialLR fallback
    use_cosine_lr: bool = True
    batch_size: int = 16
    max_epoch: int = 30
    grad_clip: float = 1.0

    # ---- Stage curriculum (CDPC-v4)
    # synth: supervised bootstrap of (trunk + refiner) on synthetic warps.
    # geom : unsupervised real-pair training of (trunk + refiner) via
    #         triplet on ShareFeature + image-space photometric.
    # cdpc : freeze geometry; train (posterior_head + uncertainty_head)
    #         with EM + Kendall-Gål-het + sigma_prior + support + smooth.
    # rel  : freeze everything else; train reliability_head on detached phi.
    # full : end-to-end ablation (everything trains together).
    stage: str = "geom"
    init_ckpt: str = ""

    # ---- Synthetic-H (synth stage)
    synth_rho_max: int = 32
    synth_iters_per_epoch: int = 4000
    use_v3_synth: bool = True

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
    log_dir: str = "train_log_v4/logs"
    model_save_dir: str = "train_log_v4/real_models"

    # =========================================================================
    # Legacy / back-compat fields. Older configs and v3 checkpoints reference
    # these; we keep them so loading does not crash. They are not consumed by
    # the v4 forward pass.
    # =========================================================================
    backbone_pretrained: bool = False
    bb_quarter_channels: int = 64
    bb_eighth_channels:  int = 128
    bb_sixteenth_channels: int = 256
    corr_radius: int = 4
    corr_out_channels: int = 32
    homography_levels: int = 3
    rho_per_level: Tuple[float, float, float] = (32.0, 16.0, 8.0)
    homography_rho: float = 32.0
    q_min_floor: float = 0.25
    lambda_align: float = 1.0
    lambda_align_soft: float = 0.5
    lambda_sigma_reg: float = 0.1
    lambda_photo: float = 0.0
    lr_h: float = 2.0e-4
    lr_backbone: float = 2.0e-5
    use_cosine_align_soft: bool = False
    use_q_weighted_photo_img: bool = False
    detach_head_inputs: bool = True          # v4 always detaches; kept for arg parsing
    log_sigma_min: float = -3.0
    log_sigma_max: float = 5.0
