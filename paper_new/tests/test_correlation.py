import torch

from paper_new.modules.correlation import build_offsets, local_correlation, soft_argmax_corr


def test_local_correlation_shape():
    feat_a = torch.randn(2, 8, 6, 7)
    feat_b = torch.randn(2, 8, 6, 7)
    center = torch.stack(torch.meshgrid(torch.arange(6), torch.arange(7), indexing="ij"), dim=-1)
    center = center[..., [1, 0]].reshape(1, -1, 2).float().repeat(2, 1, 1)
    corr = local_correlation(feat_a, feat_b, center, radius=2)
    assert corr.shape == (2, 6 * 7, 25)


def test_soft_argmax_center():
    radius = 2
    offsets = build_offsets(radius)
    center_idx = torch.nonzero((offsets == 0).all(dim=1), as_tuple=False).item()
    corr = torch.zeros(1, 3, offsets.shape[0])
    corr[..., center_idx] = 10.0
    out = soft_argmax_corr(corr, radius=radius, temperature=0.1)
    assert torch.allclose(out["delta"], torch.zeros_like(out["delta"]), atol=1e-4)
    assert out["peak"].min() > 0.99

