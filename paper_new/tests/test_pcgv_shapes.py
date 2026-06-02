import torch

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

