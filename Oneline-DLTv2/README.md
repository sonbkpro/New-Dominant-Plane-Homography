# Oneline-DLTv2 — Calibrated Dominant-Plane Consensus Homography Estimation

Implementation of `planv1.txt`. One-direction prediction of `H_ab`; predicts a
dominant-plane posterior `q_i`, spatial uncertainty `log σ_i`, and pair-level
reliability `s_ab`. Trained unsupervised on the CA-Homography pair list inherited
from v1.

## Layout

```
Oneline-DLTv2/
├── train.py                 # main training loop, with periodic eval-L2 monitoring
├── eval.py                  # full trustworthiness eval (L2 / AUROC / AUPRC / ECE / RC)
├── test.py                  # legacy point-reprojection test (matches v1's protocol)
├── configs/default.py       # all hyperparameters
├── model/
│   ├── backbone.py          # multi-scale ResNet-18 (ImageNet pretrained, 1-ch adapted)
│   ├── correlation.py       # local correlation block at 1/8
│   ├── heads.py             # homography, posterior, uncertainty, reliability heads
│   └── cdpc_net.py          # top-level CDPC network
├── losses/
│   ├── triplet.py           # L_triplet (q-weighted hinge)        — essential
│   ├── align.py             # L_align  (Kendall-Gal q-weighted)   — essential
│   ├── em.py                # L_em     (E-step posterior target)  — essential
│   ├── support.py           # L_support (q̄ floor)                 — essential
│   ├── smooth.py            # L_smooth (edge-aware TV on q)       — small weight
│   ├── reliability.py       # L_rel    (BCE + hard-negative mining)— essential
│   └── cycle.py             # L_cycle  (analytic-inverse cycle)   — optional
├── data/
│   ├── pairs.py             # natural-pair dataset (matches v1)
│   ├── invalid_pairs.py     # intra-batch shuffle + patch-reshuffle wrappers
│   └── test_dataset.py      # eval dataset with manual point correspondences
└── utils/
    ├── dlt.py               # differentiable 4-pt DLT (ported from v1)
    ├── warping.py           # F.grid_sample-based warping wrapper
    ├── inverse.py           # safe analytic H^{-1} with condition-number guard
    ├── eval_metrics.py      # L2 reprojection, AUROC, AUPRC, ECE, RC
    └── viz.py               # tensorboard visualization
```

## Setup

```bash
pip install -r requirements.txt
```

Data layout (unchanged from v1):

```
DeepHomography/Data/
├── Train_List.txt
├── Train/<video_id>/<frame>.jpg
├── Test_List.txt
├── Test/<video_id>/<frame>.jpg
└── Coordinate/<pair_id>.npy
```

## Train

```bash
cd Oneline-DLTv2
python train.py --batch_size 16 --max_epoch 30 --lr 1e-4 --amp
```

Eval-L2 on the test set runs every `--eval_every` iterations (default 2000) and
is logged to TensorBoard alongside every loss term and the reliability AUROC.

## Eval (full trustworthiness protocol)

```bash
python eval.py --ckpt train_log_v2/real_models/<ckpt>.pth
```

Reports per-category L2 reprojection (mean / median / AUC@1,3,5 / inlier@1,3,5),
plus AUROC, AUPRC, ECE, NLL, and the risk-coverage curve for failure detection.

## Legacy point-reprojection test (v1-compatible)

```bash
python test.py --ckpt train_log_v2/real_models/<ckpt>.pth
```
