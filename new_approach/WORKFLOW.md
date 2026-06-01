# No-Mask Baseline — Design & Workflow

Unsupervised deep homography baseline built **only** from the two ported
HomoGAN modules (`modules/featureHomo.py`, `modules/transformerHomo.py`) plus the
`Data/` dataset. No content-aware mask (the mask is the identity / all-ones map),
trained with an **alignment** loss + a **Feature Identity Loss (FIL)** in the
spirit of JirongZhang/DeepHomography.

---

## 1. Module analysis (what the two files actually provide)

### `featureHomo.py`
| symbol | role |
|---|---|
| `feature_extractor(in_ch, out_ch)` | **shallow** 3-conv net, channels `[in//2, 4, 8, out]`. Used as `fea_extra = feature_extractor(2, 1)`: maps a **1-ch grayscale image → 1-ch feature map** at full resolution. This is the DeepHomography `ShareFeature` analog — the space where alignment + FIL are measured. |
| `FeatureExtractor(embed_dim, num_layers)` | **strided pyramid** (each level ½ res). Used *inside* `SwinTransformer` for coarse-to-fine warping; never standalone. |

### `transformerHomo.py`
| symbol | role |
|---|---|
| `gen_basis(h,w)` | 8 QR-orthonormalized **homography-flow bases** (affine 6 + 2 quadratic terms). A homography is represented as `flow = Σ basisᵢ·weightᵢ`, a dense `(2,h,w)` field — **not** a 3×3 matrix. |
| `SwinTransformer.forward(x:(B,2,H,W))` | coarse→fine: at each pyramid level warp img2 toward img1 by the current flow, run windowed self-attention + cross-attention query tokens, regress **incremental 8 basis coeffs**. Returns `weight:(B,8,1)`. |
| `HomoNet.forward(data_batch)` | full wrapper: `fea_extra` features → `SwinTransformer` **bidirectionally** (`weight_f`, `weight_b`) → `H_flow_f`/`H_flow_b` → warps full images & features. Runs the mask predictor **only when `params.pretrain_phase=False`**. Returns every tensor the losses need. |
| `Ms_Transformer(params=…)` | factory → `HomoNet(backbone=SwinTransformer)`. |

> **Key insight:** `HomoNet` with `pretrain_phase=True` *is already* the no-mask
> baseline. We reuse it as-is (`model_baseline.build_baseline`) rather than
> rewriting the forward.

---

## 2. End-to-end data flow

```
 Train_List.txt (442k pairs)                                  Coordinate-v2 (.npy, eval only)
        │                                                              │
        ▼  dataset_baseline.TrainDataset                               │
 read pair → resize 640×360 → per-ch mean/std → grayscale              │
 full I1,I2 (2,360,640)   ;   random crop → patch (2,320,512)          │
 emit data_batch{ imgs_gray_full, imgs_gray_patch, start(2,1,1), pts } │
        │                                                              │
        ▼  model_baseline.build_baseline → HomoNet(pretrain_phase=True)│
 fea_extra (1-ch CNN) ──► f1,f2 for patch & full                       │
        │                                                              │
 SwinTransformer( cat[f1_patch, f2_patch] ) → weight_f                 │
 SwinTransformer( cat[f2_patch, f1_patch] ) → weight_b   (bidirectional)│
        │  basis ⊗ weight → H_flow_f , H_flow_b  (B,2,320,512)         │
        ▼  warp full feats/imgs by flow  (+start offset → full coords) │
   img1_patch_fea      = f(I1)                                         │
   warp_img2_patch_fea = warp(f(I2))      → aligned to I1   (Fa' fwd)  │
   img2_patch_warp_fea = f(warp(I2))                        (FIL fwd)  │
   …and the symmetric backward tensors                                 │
        │                                                              │
        ▼  losses.baseline_loss      (mask == identity)               │
   L_align = triplet(I1)  + triplet(I2)        [DeepHomography form]   │
   L_fil   = |warp(f)-f(warp)|_f + …_b                                 │
   L = λ_align·L_align + λ_fil·L_fil                                   │
        │  AdamW + grad-clip + (opt) AMP                               ▼
        ▼                                              evaluate.evaluate
 train_baseline.py ───── state_dict ckpt ─────►  eval mode: HomoNet returns
                                                 flow_f_patch at crop res (320,512,2)
                                                 geometry.flow_to_homography:
                                                   crop corners + start + flow → DLT → H_ab(3×3)
                                                 PME = mean‖H·p1 − p2‖ over first 6 points, min(LR,RL)
                                                 bucket → RE / LT / LL / SF / LF / AVG
```

---

## 3. Loss derivation (grounded in DeepHomography)

With the mask all-ones, DeepHomography's mask-weighted triplet collapses to a
mean over all pixels:

* **Alignment (triplet)** — per direction, with `Fb`=target, `Fa`=source,
  `Fa'`=warped source:
  `L_align = mean( relu(1 + |Fb − Fa'| − |Fb − Fa|) )`.
  The `+1` margin and the `−|Fb − Fa|` term prevent the degenerate
  constant-feature solution that a plain L1 on `(Fb, Fa')` would allow.
* **Feature Identity Loss (FIL)** — `mean(|warp(f(I)) − f(warp(I))|)`, forcing
  the feature extractor to commute with warping so features stay geometrically
  meaningful. Bidirectional.

Both terms read straight from the `HomoNet.forward` output dict — see
`losses.py` for the exact tensor↔term mapping.

---

## 4. Files

| file | responsibility |
|---|---|
| `model_baseline.py` | `make_params` (defaults mirror `checkpoints/stageA_v2_fc32_nocorr/args.json`) + `build_baseline` → no-mask bidirectional `HomoNet`. |
| `dataset_baseline.py` | `TrainDataset` / `TestDataset` emitting the `data_batch` dict; reuses Oneline-DLTv1's resize/grayscale/normalize. |
| `losses.py` | `triplet_align` + `baseline_loss` (align + FIL, both directions). |
| `geometry.py` | `flow_to_homography` (crop-corner DLT with crop origin) + `geometric_distance`. |
| `train_baseline.py` | training loop (AdamW, grad-clip, AMP, periodic eval, checkpointing). |
| `evaluate.py` | PME with the 5 scene-category buckets and first-6-point protocol from `Oneline-DLTv1/test.py`; `--num_points all` is available for analysis. |

---

## 5. How to run (from project root)

```sh
# Smoke (a few items/steps; ~5 s on GPU) — verifies forward/backward/eval
python -m new_approach.train_baseline \
    --max_train_items 16 --batch_size 2 --epochs 1 --max_steps 4 \
    --num_workers 0 --out_dir new_approach/checkpoints/baseline_smoke --eval_max_items 6

# Full training
python -m new_approach.train_baseline --epochs 6 --batch_size 16 --lr 4e-4 \
    --num_workers 8 --amp --out_dir new_approach/checkpoints/baseline

# Standalone evaluation of a checkpoint
python -m new_approach.evaluate --ckpt new_approach/checkpoints/baseline/baseline_final.pth
```

Environment: conda env `cv` (py3.10, torch 2.5.1+cu118); requires `timm`,
`opencv-python(-headless)`, `numpy`. CUDA is used by default; force CPU with
`CUDA_VISIBLE_DEVICES=""`.

---

## 6. Notes / next steps toward `plan.txt`

* The network predicts an **8-DOF homography *flow*** on the cropped patch; the
  3×3 `H` is recovered at eval from the four crop corners in full-image
  coordinates. The flow basis includes 2 quadratic terms, so `H` is the
  DLT-best-fit homography to that crop flow.
* Bidirectionality (`H_flow_f`/`H_flow_b`) is already computed — an
  **inverse-consistency** term (`H_ab·H_ba ≈ I`) is the natural next loss to add.
* Re-enabling the mask (`pretrain_phase=False`) activates HomoNet's mask
  predictor + needs the `Discriminator`; that is the dominant-plane / GAN stage,
  out of scope for this baseline.
```
