"""Baseline homography model: featureHomo + transformerHomo, no mask.

Builds the HomoGAN `HomoNet` in its `pretrain_phase=True` configuration, which
disables the content-aware mask predictor (the mask is effectively the identity
/ all-ones map). The network therefore reduces to:

    fea_extra (featureHomo.feature_extractor)  ->  1-ch feature maps
    SwinTransformer (transformerHomo)          ->  8 homography-flow basis coeffs
    basis (x) coeffs                           ->  dense bidirectional flow H_flow_f/H_flow_b

This is the unsupervised baseline for the plan.txt extension. See
`new_approach/WORKFLOW.md` for the full data-flow description.
"""
from types import SimpleNamespace

try:
    from .modules.transformerHomo import Ms_Transformer
except ImportError as exc:
    if __package__:
        raise
    # Allow direct execution from inside new_approach/ as a flat script.
    from modules.transformerHomo import Ms_Transformer

# Full frame / patch geometry (matches the stageA_* checkpoint args.json).
FULL_H, FULL_W = 360, 640
CROP_H, CROP_W = 320, 512


def make_params(crop_h=CROP_H, crop_w=CROP_W, pretrain_phase=True, **overrides):
    """Assemble the attribute bag consumed by HomoNet / SwinTransformer.

    `pretrain_phase=True` is the no-mask baseline path inside HomoNet.forward.
    Defaults mirror new_approach/checkpoints/stageA_v2_fc32_nocorr/args.json.
    """
    params = SimpleNamespace(
        # --- geometry ---
        crop_size=(crop_h, crop_w),
        # --- featureHomo.feature_extractor (fea_extra): expects 1-ch image in ---
        in_channels=2,            # channels[0] = in_channels // 2 = 1
        # --- transformer (SwinTransformer) ---
        num_basis=8,
        embed_dim=48,
        depths=[2, 4, 6],
        layer_depth=[3, 2, 1],
        num_heads=[4, 8, 16],
        num_decoder_layers=3,
        window_size=8,
        patch_size=4,
        in_chans=2,               # PatchEmbed (constructed but unused in forward)
        mlp_ratio=3.0,
        qkv_bias=True,
        qk_scale=None,
        ape=False,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        # --- mask / phase ---
        pretrain_phase=pretrain_phase,
        mask_use_fea=False,
        net_type="HomoGAN",
    )
    for k, v in overrides.items():
        setattr(params, k, v)
    return params


def build_baseline(params=None):
    """Construct the bidirectional, no-mask HomoNet baseline."""
    if params is None:
        params = make_params()
    return Ms_Transformer(pretrained=False, params=params)
