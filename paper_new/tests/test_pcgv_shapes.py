import torch

from paper_new.geometry import torch_warp_full_with_patch_flow
from paper_new.model_pcgv import build_pcgv, make_pcgv_params
from paper_new.modules.pcgv import PCGVModule


def test_pcgv_forward_shape():
    torch.manual_seed(7)
    feat_a = torch.randn(2, 16, 8, 10)
    feat_b = feat_a + 0.01 * torch.randn_like(feat_a)
    pcgv = PCGVModule(
        feat_dim=16,
        hidden_dim=32,
        num_iters=2,
        radius=2,
        use_plane_token=True,
        use_uncertainty=True,
    )
    out = pcgv(feat_a, feat_b)
    assert out["H"].shape == (2, 3, 3)
    assert out["mask"].shape == (2, 1, 8, 10)
    assert out["votes"].shape == (2, 80, 1)
    assert out["matches"].shape == (2, 80, 2)
    assert out["grid"].shape == (2, 80, 2)
    assert out["uncertainty"].shape == (2, 1, 8, 10)


def test_no_nan_forward():
    feat_a = torch.randn(1, 8, 6, 6)
    feat_b = torch.randn(1, 8, 6, 6)
    pcgv = PCGVModule(feat_dim=8, hidden_dim=16, num_iters=2, radius=1)
    out = pcgv(feat_a, feat_b)
    for key in ("H", "mask", "votes", "matches", "residuals", "uncertainty"):
        assert torch.isfinite(out[key]).all(), key


def test_full_frame_patch_flow_uses_crop_start():
    src = torch.arange(1 * 1 * 5 * 6, dtype=torch.float32).reshape(1, 1, 5, 6)
    flow = torch.zeros(1, 2, 3, 2)
    start = torch.tensor([[[[2.0]], [[1.0]]]])
    warped = torch_warp_full_with_patch_flow(src, flow, start)
    expected = src[:, :, 1:3, 2:5]
    assert torch.equal(warped, expected)


def test_pcgv_features_can_initialize_from_coarse():
    params = make_pcgv_params(
        crop_h=64,
        crop_w=64,
        pcgv_feat_dim=8,
        pcgv_hidden_dim=16,
        pcgv_num_iters=1,
        pcgv_radius=1,
        pcgv_pyramid_embed_dim=24,
    )
    net = build_pcgv(params)
    report = net.init_pcgv_features_from_coarse()
    assert report["shallow_copied"] > 0
    assert report["pyramid_copied"] > 0
