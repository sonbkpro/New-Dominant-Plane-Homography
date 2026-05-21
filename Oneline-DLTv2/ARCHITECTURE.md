# Oneline-DLTv2 Architecture

![Oneline-DLTv2 architecture diagram](architecture_diagram.svg)

This is the complete data flow implemented by `model/cdpc_net.py` and trained by
`train.py`. The central idea is:

- Use a joint 2-channel path to regress one global homography `H_ab`.
- Use separate per-image features to estimate dominant-plane support `q`,
  uncertainty `log_sigma`, residuals, and pair reliability `s`.
- Train everything with feature alignment, posterior calibration, reliability,
  support, smoothness, and optional cycle losses.

## Forward Pass

```mermaid
flowchart TB
    Ia["I_a patch<br/>(B,1,315,560)"]
    Ib["I_b patch<br/>(B,1,315,560)"]

    subgraph PerImage["Per-image feature path"]
        Ba["MultiScaleBackbone<br/>ResNet-18 stem + layer1 + layer2"]
        Bb["MultiScaleBackbone<br/>same weights as path A"]
        Fa4["F_a4<br/>(B,64,H/4,W/4)"]
        Fa8["F_a8<br/>(B,128,H/8,W/8)"]
        Fb4["F_b4<br/>(B,64,H/4,W/4)"]
        Fb8["F_b8<br/>(B,128,H/8,W/8)"]
    end

    subgraph Joint["Joint homography path"]
        Cat2["cat(I_a,I_b)<br/>(B,2,H,W)"]
        JB["Joint MultiScaleBackbone<br/>2-channel ResNet-18"]
        Fj8["F_joint8<br/>(B,128,H/8,W/8)"]
    end

    subgraph Corr["Correspondence feature"]
        LC["LocalCorrelation R=4<br/>cosine cost volume: 81 channels"]
        C8["c8<br/>(B,32,H/8,W/8)"]
        C4["upsample c8 to 1/4<br/>c4"]
    end

    subgraph HPath["Homography regression"]
        HIn["cat(F_joint8,c8)<br/>(B,160,H/8,W/8)"]
        HHead["HomographyHead<br/>3 conv blocks + GAP + MLP"]
        Off["offset_ab<br/>(B,8) corner offsets"]
        DLT["DLT_solve<br/>patch corners + offsets"]
        Hab["H_ab<br/>(B,3,3), patch coords"]
        Hq["rescale to 1/4 coords<br/>H_quarter"]
    end

    subgraph WarpResidual["Warp and residual"]
        Warp["warp F_a4 by H_quarter<br/>grid_sample"]
        Faw["F_a_warped<br/>(B,64,H/4,W/4)"]
        Valid["valid_mask<br/>(B,1,H/4,W/4)"]
        Res["residual r<br/>mean_c abs(F_b4 - F_a_warped)"]
    end

    subgraph PixelHeads["Pixel heads at 1/4 scale"]
        PostIn["cat(F_b4,F_a_warped,c4,r)<br/>(B,161,H/4,W/4)"]
        QHead["PosteriorHead<br/>conv tower + sigmoid"]
        SHead["UncertaintyHead<br/>conv tower + clamp"]
        Q["q_ab = q * valid_mask<br/>(B,1,H/4,W/4)"]
        LogSig["log_sigma_ab<br/>(B,1,H/4,W/4)"]
    end

    subgraph Reliability["Reliability branch"]
        Inv["safe_inverse_3x3(H_quarter)"]
        CycleWarp["warp F_a_warped by H_quarter^-1"]
        Farec["F_a_recovered"]
        Phi["phi = 8 pooled stats<br/>q_mean,q_var,q_area,r_mean,<br/>q*r_mean,sigma_mean,cycle_mean,offset_norm"]
        RelHead["ReliabilityHead<br/>MLP + sigmoid"]
        Score["s<br/>(B,)"]
    end

    Ia --> Ba
    Ib --> Bb
    Ba --> Fa4
    Ba --> Fa8
    Bb --> Fb4
    Bb --> Fb8

    Ia --> Cat2
    Ib --> Cat2
    Cat2 --> JB --> Fj8

    Fa8 --> LC
    Fb8 --> LC
    LC --> C8
    C8 --> C4

    Fj8 --> HIn
    C8 --> HIn
    HIn --> HHead --> Off --> DLT --> Hab --> Hq

    Fa4 --> Warp
    Hq --> Warp
    Warp --> Faw
    Hq --> Valid
    Fb4 --> Res
    Faw --> Res

    Fb4 --> PostIn
    Faw --> PostIn
    C4 --> PostIn
    Res --> PostIn
    PostIn --> QHead --> Q
    PostIn --> SHead --> LogSig
    Valid --> Q

    Hq --> Inv --> CycleWarp
    Faw --> CycleWarp --> Farec
    Q --> Phi
    Res --> Phi
    LogSig --> Phi
    Farec --> Phi
    Off --> Phi
    Phi --> RelHead --> Score
```

## One-Screen Summary

```text
Inputs:
  I_a, I_b grayscale patches: (B, 1, 315, 560)

Backbones:
  per-image backbone(I_a) -> F_a4, F_a8
  per-image backbone(I_b) -> F_b4, F_b8
  joint backbone(cat(I_a,I_b)) -> F_joint8

Matching:
  LocalCorrelation(F_a8, F_b8) -> c8

Homography:
  HomographyHead(cat(F_joint8,c8)) -> 8 corner offsets
  DLT_solve(corners, offsets) -> H_ab

Dominant-plane reasoning:
  warp(F_a4, H_ab at 1/4 scale) -> F_a_warped
  residual = mean_channel(abs(F_b4 - F_a_warped))
  cat(F_b4, F_a_warped, upsample(c8), residual) -> q, log_sigma

Reliability:
  analytic inverse + cycle warp + pooled q/residual/sigma/offset stats -> s

Main outputs:
  H_ab, H_ab_inv, offset_ab, q_ab, log_sigma_ab, residual_map_ab,
  reliability_score, feature_a, feature_b
```

## Training Loss Wiring

```mermaid
flowchart LR
    Outputs["CDPCNet outputs"]

    subgraph Out["Forward outputs used by losses"]
        Fb4["F_b4"]
        Faw["F_a_warped"]
        Fa4["F_a4"]
        Q["q"]
        LS["log_sigma"]
        R["residual"]
        VM["valid_mask"]
        S["s"]
        Farec["F_a_recovered"]
        CV["cycle_valid"]
        Cond["cond_valid"]
    end

    LTrip["L_triplet<br/>hinge: d_pos(F_b4,F_a_warped)<br/>vs d_neg(F_b4,F_a4)"]
    LAlign["L_align<br/>q-weighted heteroscedastic<br/>Charbonnier residual"]
    LEM["L_em<br/>BCE(q, E-step target from residual/sigma)"]
    LSupport["L_support<br/>keeps mean q above alpha"]
    LSmooth["L_smooth<br/>edge-aware TV on q"]
    LRel["L_rel<br/>BCE(s, natural/invalid/hard labels)"]
    LCycle["L_cycle<br/>feature cycle consistency"]
    LSigma["L_sigma_reg<br/>keeps log_sigma near 0"]
    Total["Weighted sum -> L_total"]

    Outputs --> Fb4
    Outputs --> Faw
    Outputs --> Fa4
    Outputs --> Q
    Outputs --> LS
    Outputs --> R
    Outputs --> VM
    Outputs --> S
    Outputs --> Farec
    Outputs --> CV
    Outputs --> Cond

    Fb4 --> LTrip
    Faw --> LTrip
    Fa4 --> LTrip
    VM --> LTrip

    R --> LAlign
    LS --> LAlign
    Q --> LAlign
    VM --> LAlign

    Q --> LEM
    R --> LEM
    LS --> LEM
    VM --> LEM

    Q --> LSupport
    VM --> LSupport

    Q --> LSmooth

    S --> LRel

    Fa4 --> LCycle
    Farec --> LCycle
    CV --> LCycle
    Cond --> LCycle

    LS --> LSigma
    VM --> LSigma

    LTrip --> Total
    LAlign --> Total
    LEM --> Total
    LSupport --> Total
    LSmooth --> Total
    LRel --> Total
    LCycle --> Total
    LSigma --> Total
```

## Why There Are Two Backbone Routes

The joint route is for `H_ab`. It sees `cat(I_a, I_b)` from the first
convolution onward, so the homography regressor can learn pairwise cues early.
This route outputs only `F_joint8`, which is concatenated with the local
correlation feature `c8` before the homography head.

The per-image route is for calibrated consensus. It keeps `F_a` and `F_b`
separate, which makes the residual map meaningful after warping:

```text
residual(i) = mean_c |F_b4(i) - warp(F_a4, H_ab)(i)|
```

That residual, together with warped/source features and correlation, drives:

- `q`: pixelwise dominant-plane posterior.
- `log_sigma`: pixelwise uncertainty for alignment.
- `s`: pair-level reliability score from pooled statistics.

## Spatial Scales

| Tensor | Scale | Default shape |
| --- | --- | --- |
| `I_a`, `I_b` | full patch | `(B,1,315,560)` |
| `F_a4`, `F_b4` | 1/4 | `(B,64,79,140)` |
| `F_a8`, `F_b8`, `F_joint8` | 1/8 | `(B,128,40,70)` |
| `c8` | 1/8 | `(B,32,40,70)` |
| `c4`, `q`, `log_sigma`, `residual` | 1/4 | `(B,*,79,140)` |
| `offset_ab` | global | `(B,8)` |
| `H_ab` | global | `(B,3,3)` |
| `s` | pair-level | `(B,)` |

The 1/4 and 1/8 sizes come from the ResNet stem/maxpool/layer strides and
padding.

## Evaluation Path

During evaluation, the network predicts `H_ab` in patch coordinates. `train.py`
and `eval.py` convert it back to full-image coordinates with:

```text
H_full = T_crop * H_patch * T_crop^-1
```

Then the code measures point reprojection error on the manual correspondences.

## File Map

- `model/cdpc_net.py`: top-level architecture and output dictionary.
- `model/backbone.py`: ResNet-18 multi-scale feature extractor.
- `model/correlation.py`: local cosine correlation block.
- `model/heads.py`: homography, posterior, uncertainty, reliability heads.
- `utils/dlt.py`: differentiable 4-point DLT.
- `utils/warping.py`: homography warping and validity masks.
- `losses/*.py`: individual training objectives.
- `train.py`: dataset wiring, invalid-pair construction, loss weighting,
  AMP, logging, checkpointing, and periodic eval-L2.
