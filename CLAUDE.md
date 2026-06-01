# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

PyTorch implementation of unsupervised deep homography estimation, forked from the ECCV 2020 paper *"Content-Aware Unsupervised Deep Homography Estimation"* (JirongZhang/DeepHomography). The active code is the simplified **Oneline-DLT** variant: it directly predicts a single forward homography `H_ab` from a stacked grayscale image pair and trains with a mask-weighted triplet feature loss (no ground-truth labels).

`plan.txt` is the research roadmap. It plans to extend this baseline into *"Calibrated Dominant-Plane Consensus for Trustworthy Unsupervised Deep Homography Estimation"* (target: IEEE TIP) — adding bidirectional `H_ab`/`H_ba` with inverse consistency, a dominant-plane posterior (replacing the attention mask), spatial uncertainty, and a pair-level reliability score. When working toward those features, treat `plan.txt` §5–7 as the spec and the current `Oneline-DLTv1/` code as the baseline to build on. (`plan.txt` is git-tracked but may be absent/deleted in a given working tree — see "Branches" below.)

`new_approach/` is the in-progress home of that extension; it is a **different, HomoGAN-derived architecture**, not a modification of `Oneline-DLTv1/`. See its section below.

## Branches

This repo has several long-lived divergent feature branches (`Deverlop1`…`Deverlop4`, plus `master`); they differ substantially in what `new_approach/` and the root-level `plan*.txt` files contain. **Check `git branch` and the actual working-tree contents before assuming a file exists** — e.g. `plan.txt` is tracked but currently deleted on `Deverlop4`, and `new_approach/`'s trainer/geometry/loss `.py` sources are present on some branches and missing on others (only stale `.pyc` bytecode and leftover checkpoints remain). When something referenced here or in `plan.txt` isn't on disk, it likely lives on a sibling branch rather than being genuinely gone.

## Commands

All scripts live in `Oneline-DLTv1/` and **must be run from inside that directory.** They use flat imports (`import resnet`, `from dataset import ...`) and compute the project root as the *parent of the current working directory* via `os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))`. Note `os.path.dirname("__file__")` is a literal-string quirk that returns `''`, so the resolved root is `<cwd>/..` — running from anywhere other than `Oneline-DLTv1/` silently points the data/model paths at the wrong place.

```sh
cd Oneline-DLTv1

# Train from scratch
python train.py --gpus 2 --cpus 8 --lr 0.0001 --batch_size 32

# Two-stage finetune (see "Two-stage training" below)
python train.py --gpus 2 --cpus 8 --lr 0.000064 --batch_size 32 --finetune True

# Evaluate (defaults to ../train_log_Oneline-FastDLT/real_models/resnet34_iter_44000.pth)
python test.py
python test.py --model_path ../train_log_Oneline-FastDLT/real_models/<ckpt>.pth

# Visualize one pair's alignment + learned mask
python visualization_align.py \
    --npy ../Data/Coordinate-v2/00000100_10001.jpg_00000100_10005.jpg.npy \
    --model_path ../train_log_Oneline-FastDLT/real_models/resnet34_iter_44000.pth
```

Data prep (run from `Data/`, requires the downloaded raw videos in `Data/Train/` and `Data/Test/`):
```sh
cd Data && python video2img.py   # extracts video frames to per-video subfolders
```

There is no test suite, linter, or build step — this is a research codebase. "Testing" means `test.py` (reprojection-error evaluation).

### Arg semantics worth knowing
- `--cpus` → DataLoader `num_workers`. `--gpus` is effectively unused for device placement; training wraps the net in `torch.nn.DataParallel` over all visible GPUs (control with `CUDA_VISIBLE_DEVICES`).
- `--pretrained` loads an ImageNet ResNet backbone but **excludes** `conv1`/`fc` (their shapes differ — see below).
- `--finetune True` in `train.py` loads `../models/freeze-mask-first-fintune.pth`; in `test.py` it means "load a trained checkpoint" and is on by default.

## Runtime environment

README lists Python 3.6 / PyTorch 1.0.1, but the code actually runs on **Python 3.10 + a modern PyTorch** (compiled `.cpython-310` artifacts; `test.py` uses `torch.load(..., weights_only=False)`; `scheduler.get_last_lr()`). Match that when setting up an environment. Requires `torch`, `torchvision`, `opencv-python`, `numpy`, `tensorboardX`, `imageio`.

## Architecture

The whole pipeline is a single customized `ResNet.forward` in [resnet.py](Oneline-DLTv1/resnet.py); warping is done inside `forward` (not the dataset) deliberately, to balance load under `DataParallel`. Three sub-networks cooperate:

1. **Backbone** (`conv1`…`fc`): a ResNet (default resnet34) modified so `conv1` accepts **2 input channels** (the stacked grayscale patch pair) and `fc` outputs **8 values** — the 4-corner offset `Δp` representation. Backbone construction/head-swapping is in [torch_homography_model.py](Oneline-DLTv1/torch_homography_model.py) `build_model`.
2. **`ShareFeature`**: a small CNN learning a 1-channel feature map `f(·)`. The triplet loss is computed in this feature space, not raw pixels.
3. **`genMask`**: a small CNN ending in sigmoid, producing the content-aware mask `m(·)` for outlier suppression.

Forward flow: `genMask` runs on the full images → patches are gathered (`getPatchFromFullimg`) and normalized (`normMask`) → `ShareFeature` extracts features from the input patches → features are multiplied by the mask and concatenated → backbone → 8-dim corner offset → `DLT_solve` builds `H_mat` → `transform` warps image 1's features toward image 2 → mask-weighted hinge/triplet loss:
`feature_loss = Σ(clamp(1 + |Fb − Fa'| − |Fb − Fa|, 0) · mask_ap) / Σ(mask_ap)`.

`DLT_solve` and `transformer`/`transform` (the differentiable spatial-transformer warp) live in [utils.py](Oneline-DLTv1/utils.py). `DLT_solve` is written to support a multi-homography mesh (`divide×divide` grid), but Oneline uses the degenerate 1×1 case (single global H). `getBatchHLoss` (inverse-consistency `H·H⁻¹≈I`) exists but is unused by the current Oneline loss — it's groundwork for the `plan.txt` bidirectional extension.

### Two-stage training (mask freezing)
For stable convergence, train in two stages by toggling **one line** in [resnet.py:280](Oneline-DLTv1/resnet.py#L280):
- **Stage 1:** uncomment `mask_ap = torch.ones_like(mask_ap)` so the mask is all-ones and its gradient doesn't update — lets `ShareFeature` learn stable features first (≥2 epochs).
- **Stage 2:** re-comment that line and finetune at a lower LR so `genMask` starts learning.

### Checkpoints
`train.py` saves with `torch.save(net, ...)` — the **entire `DataParallel` model object**, not a `state_dict` — every `model_save_fre` (4000) iterations to `train_log_Oneline-FastDLT/real_models/`. `test.py`'s `load_model_weights` defensively unwraps `DataParallel`/`Module`/`dict` checkpoints and strips `module.` prefixes, so it handles any of these formats.

## Data layout

- `Data/Train_List.txt`, `Test_List.txt`, `Val_List.txt` — committed. Each line is a space-separated image pair, e.g. `000001/000001_10001.jpg 000001/000001_10004.jpg` (`<video_id>/<frame>.jpg`).
- `Data/Train/`, `Data/Test/` — extracted frames in per-video subfolders. **Not committed** (gitignored); must be produced from the downloaded dataset via `video2img.py`. See README for the download links.
- `Data/Coordinate/`, `Data/Coordinate-v2/` — `.npy` files of 6 manually labeled point correspondences per test pair, used **only for evaluation**. `Coordinate-v2` is the more accurate release. (`Coordinate/` is currently empty here.)
- Images are resized to 640×360, converted to grayscale, normalized by a fixed mean/std, then a `560×315` patch is cropped (random in training; fixed at offset `(40,23)` for testing so the patch sits mid-frame).

### Evaluation
`test.py` reports mean reprojection error over 6 correspondence points per pair, bucketed into 5 scene categories by hardcoded video-id lists: **RE** (regular), **LT** (low texture), **LL** (low light), **SF** (small foreground), **LF** (large foreground). `geometricDistance` takes `min(err_LR, err_RL)` because the human annotator didn't fix a consistent left/right point ordering. Results and alignment GIFs are written to `exp_result_Oneline-FastDLT/`.

## `new_approach/` — the plan.txt extension (HomoGAN-derived, WIP)

A separate, transformer-based reimplementation pursuing the `plan.txt` goals — **not** a fork of `Oneline-DLTv1/`. Unlike the Oneline scripts, its tooling is run **from the project root** as a module/path (e.g. `python new_approach/_diag_bn.py`), and config is captured per-run as `new_approach/checkpoints/<run>/args.json` rather than passed as a few CLI flags. Datasets/lists are reused from `Data/` (`Data/Coordinate-v2/Coordinate-v2`, `Train_List.txt`, etc.).

What's source-present on this branch (the rest exists only on sibling branches — see "Branches"):
- `modules/transformerHomo.py` — verbatim-ish port of megvii-research/HomoGAN's `model/swin_multi.py` + `model/net.py`. Swin-transformer homography stack: `SwinTransformer`, `HomoNet`, `Discriminator`, cross-attention blocks (`WindowCrossAttention`/`SwinCrossBlock`), basis generation (`gen_basis`), and a flow-warp `transformer`. Built via the `Ms_Transformer(pretrained, **kwargs)` factory.
- `modules/featureHomo.py` — HomoGAN's multi-scale `FeatureExtractor` pyramid + a `feature_extractor(...)` conv helper.

The training pipeline (mode `pretrain_homo`) is driven by an entry script (`train_new.py`) plus `geometry_homo.py`, `losses_homo.py`, `modules/maskFlowHomo.py`, `modules/correlation.py`, `evaluate.py`, and `Data/homo_flow_dataset.py` — referenced by `_diag_bn.py`, the checkpoint `args.json`, and `__pycache__/` bytecode, but **their `.py` sources are not checked out on this branch**. The committed `args.json` files document the intended hyperparameters: a Swin encoder (`embed_dim`, `depths`, `num_heads`, `window_size`) with iterative refinement (`refine_iters`), an optional correlation volume (`use_correlation`/`corr_radius`), and a flow-based mask head (`mask_method: homogan_cnn`, the `mask_flow_*` knobs) trained with a multi-term loss (`lambda_align`/`fm`/`mask_bce`/`dice`/`area`/`tv`/`entropy`/`fg`) and robust alignment (`align_robust: truncated`/`tukey`). `_diag_bn.py` is a throwaway diagnostic for a BatchNorm train/eval-mismatch bug (delete after use).
