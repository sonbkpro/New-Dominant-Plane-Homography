import torch
import torch.nn.functional as F

from paper_new.geometry import torch_warp_full_with_patch_flow
from paper_new.losses_pcgv import dominant_plane_loss, robust_vote_pseudo_label_loss
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
    assert out["H_refined"].shape == (2, 3, 3)
    assert out["H_start"].shape == (2, 3, 3)
    assert out["mask"].shape == (2, 1, 8, 10)
    assert out["votes"].shape == (2, 80, 1)
    assert out["matches"].shape == (2, 80, 2)
    assert out["grid"].shape == (2, 80, 2)
    assert out["uncertainty"].shape == (2, 1, 8, 10)
    assert out["refine_blend"].ndim == 0
    assert out["dlt_fallbacks"].ndim == 0
    assert out["dlt_attempts"].ndim == 0


def test_no_nan_forward():
    feat_a = torch.randn(1, 8, 6, 6)
    feat_b = torch.randn(1, 8, 6, 6)
    pcgv = PCGVModule(feat_dim=8, hidden_dim=16, num_iters=2, radius=1)
    applied_blend = pcgv.set_refine_blend(0.08)
    assert abs(applied_blend - 0.08) < 1e-6
    assert abs(pcgv.get_refine_blend() - 0.08) < 1e-6
    out = pcgv(feat_a, feat_b)
    for key in ("H", "H_refined", "H_start", "mask", "votes", "matches", "residuals", "uncertainty", "refine_blend"):
        assert torch.isfinite(out[key]).all(), key
    assert torch.isfinite(out["dlt_fallbacks"]).all()


def test_robust_vote_loss_smoke():
    pcgv = PCGVModule(feat_dim=8, hidden_dim=16, num_iters=2, radius=1)
    feat_a = torch.randn(1, 8, 6, 6)
    feat_b = torch.randn(1, 8, 6, 6)
    pcgv_out = pcgv(feat_a, feat_b)
    out = {
        "pcgv_votes_f": pcgv_out["votes"],
        "pcgv_residuals_f": pcgv_out["residuals"],
        "pcgv_corr_stats_f": pcgv_out["stats"],
    }
    loss, logs = robust_vote_pseudo_label_loss(out)
    assert torch.isfinite(loss)
    assert "pseudo_hi_f" in logs
    assert "pseudo_lo_f" in logs


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


def test_bilinear_patch_mask_matches_vote_mask_when_refine_disabled():
    params = make_pcgv_params(
        crop_h=64,
        crop_w=64,
        pcgv_feat_dim=8,
        pcgv_hidden_dim=16,
        pcgv_num_iters=1,
        pcgv_radius=1,
        pcgv_mask_refine=False,
    )
    net = build_pcgv(params)
    mask = torch.rand(2, 1, 5, 7)
    expected = F.interpolate(mask, size=(17, 19), mode="bilinear", align_corners=True)
    actual = net._upsample_pcgv_mask(mask, size=(17, 19))
    assert torch.allclose(actual, expected)


def test_learned_mask_upsampler_still_available():
    params = make_pcgv_params(
        crop_h=64,
        crop_w=64,
        pcgv_feat_dim=8,
        pcgv_hidden_dim=16,
        pcgv_num_iters=1,
        pcgv_radius=1,
        pcgv_mask_refine=True,
    )
    net = build_pcgv(params)
    mask = torch.rand(2, 1, 5, 7)
    actual = net._upsample_pcgv_mask(mask, size=(17, 19))
    assert actual.shape == (2, 1, 17, 19)
    assert torch.isfinite(actual).all()


def test_dominant_plane_loss_smoke():
    torch.manual_seed(11)
    batch, channels, height, width = 2, 3, 8, 9
    out = {
        "img1_patch": torch.randn(batch, 1, height, width),
        "img2_patch": torch.randn(batch, 1, height, width),
        "img1_patch_fea": torch.randn(batch, channels, height, width),
        "img2_patch_fea": torch.randn(batch, channels, height, width),
        "warp_img2_patch_fea": torch.randn(batch, channels, height, width),
        "warp_img1_patch_fea": torch.randn(batch, channels, height, width),
        "img2_patch_warp_fea": torch.randn(batch, channels, height, width),
        "img1_patch_warp_fea": torch.randn(batch, channels, height, width),
        "pcgv_mask_f_patch": torch.rand(batch, 1, height, width),
        "pcgv_mask_b_patch": torch.rand(batch, 1, height, width),
        "pcgv_votes_f": torch.rand(batch, height * width, 1),
        "pcgv_votes_b": torch.rand(batch, height * width, 1),
        "dlt_fallbacks_f": torch.tensor(0.0),
        "dlt_fallbacks_b": torch.tensor(0.0),
        "dlt_attempts_f": torch.tensor(4.0),
        "dlt_attempts_b": torch.tensor(4.0),
    }
    loss, logs = dominant_plane_loss(out, lambda_tv=0.01, lambda_cycle=0.0)
    assert torch.isfinite(loss)
    assert "coverage" in logs
    assert "mask_area_f" in logs
