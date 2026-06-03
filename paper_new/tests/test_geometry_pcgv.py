import math

import torch

from paper_new.geometry import (
    torch_dlt_leverage,
    torch_flow_to_homography_from_corners,
    torch_homography_to_flow,
    torch_make_pixel_grid,
    torch_warp_points_h,
    torch_weighted_dlt,
)


def _assert_h_close(H, expected, atol=1e-3):
    H = H / H[:, 2:3, 2:3]
    expected = expected / expected[:, 2:3, 2:3]
    assert torch.allclose(H, expected, atol=atol, rtol=atol)


def test_weighted_dlt_identity():
    grid = torch_make_pixel_grid(2, 8, 9, dtype=torch.float32)
    weights = torch.ones(2, grid.shape[1], 1)
    H = torch_weighted_dlt(grid, grid, weights)
    expected = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    _assert_h_close(H, expected)


def test_weighted_dlt_translation():
    grid = torch_make_pixel_grid(1, 10, 12, dtype=torch.float32)
    shift = torch.tensor([3.0, -2.0])
    dst = grid + shift
    weights = torch.ones(1, grid.shape[1], 1)
    H = torch_weighted_dlt(grid, dst, weights)
    expected = torch.tensor([[[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]])
    _assert_h_close(H, expected)


def test_weighted_dlt_random_homography():
    grid = torch_make_pixel_grid(1, 9, 11, dtype=torch.float32)
    H_true = torch.tensor(
        [[[1.02, 0.03, 2.0],
          [-0.02, 0.98, 1.5],
          [0.0005, -0.0003, 1.0]]],
        dtype=torch.float32,
    )
    dst = torch_warp_points_h(grid, H_true)
    weights = torch.ones(1, grid.shape[1], 1)
    H = torch_weighted_dlt(grid, dst, weights)
    pred = torch_warp_points_h(grid, H)
    assert (pred - dst).norm(dim=-1).mean() < 1e-3


def test_weight_zero_outliers():
    grid = torch_make_pixel_grid(1, 12, 12, dtype=torch.float32)
    dst = grid + torch.tensor([2.0, 1.0])
    dst_corrupt = dst.clone()
    dst_corrupt[:, :20] += 25.0
    weights = torch.ones(1, grid.shape[1], 1)
    weights[:, :20] = 1e-6
    H = torch_weighted_dlt(grid, dst_corrupt, weights)
    pred = torch_warp_points_h(grid[:, 20:], H)
    assert (pred - dst[:, 20:]).norm(dim=-1).mean() < 1e-2


def test_weighted_dlt_bad_input_returns_nonfinite_h():
    grid = torch_make_pixel_grid(1, 8, 8, dtype=torch.float32)
    dst = grid.clone()
    dst[:, 0, 0] = float("nan")
    weights = torch.ones(1, grid.shape[1], 1)
    H = torch_weighted_dlt(grid, dst, weights)
    assert H.shape == (1, 3, 3)
    assert not torch.isfinite(H).all()


def test_dlt_leverage_shape_finite_nonnegative():
    grid = torch_make_pixel_grid(2, 8, 9, dtype=torch.float32)
    dst = grid + torch.tensor([1.0, -0.5])
    weights = torch.ones(2, grid.shape[1], 1)
    leverage = torch_dlt_leverage(grid, dst, weights)
    assert leverage.shape == (2, grid.shape[1], 1)
    assert torch.isfinite(leverage).all()
    assert (leverage >= 0).all()
    assert torch.allclose(leverage.mean(dim=1), torch.ones(2, 1), atol=1e-3, rtol=1e-3)


def test_dlt_leverage_tolerates_near_zero_weights():
    grid = torch_make_pixel_grid(1, 7, 7, dtype=torch.float32)
    dst = grid + torch.tensor([0.25, 0.75])
    weights = torch.ones(1, grid.shape[1], 1)
    weights[:, :12] = 1e-9
    leverage = torch_dlt_leverage(grid, dst, weights)
    assert torch.isfinite(leverage).all()
    assert (leverage >= 0).all()


def test_homography_to_flow_shape_and_corner_recovery():
    H = torch.tensor([[[1.0, 0.0, 4.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]])
    flow = torch_homography_to_flow(H, 16, 20)
    assert flow.shape == (1, 16, 20, 2)
    H_rec = torch_flow_to_homography_from_corners(flow)
    _assert_h_close(H_rec, H, atol=1e-3)
