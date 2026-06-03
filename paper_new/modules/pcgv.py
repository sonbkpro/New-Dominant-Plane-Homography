"""Main Progressive Correlation-Guided Voting module."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from paper_new.geometry import (
    torch_dlt_condition_number,
    torch_dlt_leverage,
    torch_make_pixel_grid,
    torch_normalize_homography,
    torch_pixel_to_norm,
    torch_warp_points_h,
    torch_weighted_dlt,
)
from paper_new.modules.correlation import correlation_match, sample_features_at_points
from paper_new.modules.pcgv_blocks import (
    EvidenceEncoder,
    GeometryAwareTransformer,
    PlaneTokenConsensus,
    UncertaintyHead,
    VoteGRU,
    VoteHead,
)


class PCGVModule(nn.Module):
    """Iterative feature correspondence voting with weighted DLT refinement."""

    def __init__(self,
                 feat_dim: int = 64,
                 hidden_dim: int = 128,
                 num_iters: int = 4,
                 radius: int = 4,
                 temperature: float = 0.1,
                 min_vote: float = 1e-4,
                 use_transformer: bool = False,
                 use_plane_token: bool = True,
                 use_uncertainty: bool = True,
                 use_leverage: bool = True,
                 damped_update: bool = True,
                 update_alpha: float = 0.7,
                 refine_blend_init: float = 0.05,
                 learn_refine_blend: bool = True):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_iters = num_iters
        self.radius = radius
        self.temperature = temperature
        self.min_vote = min_vote
        self.use_transformer = use_transformer
        self.use_plane_token = use_plane_token
        self.use_uncertainty = use_uncertainty
        self.use_leverage = use_leverage
        self.damped_update = damped_update
        self.update_alpha = update_alpha
        blend = min(max(float(refine_blend_init), 1e-4), 1.0 - 1e-4)
        blend_logit = math.log(blend / (1.0 - blend))
        self.refine_blend_logit = nn.Parameter(
            torch.tensor(blend_logit, dtype=torch.float32),
            requires_grad=learn_refine_blend,
        )

        evidence_dim = feat_dim * 2 + 10
        self.encoder = EvidenceEncoder(evidence_dim, hidden_dim)
        self.vote_gru = VoteGRU(hidden_dim)
        self.geometry_transformer = (
            GeometryAwareTransformer(hidden_dim)
            if use_transformer else None
        )
        self.plane_token = PlaneTokenConsensus(hidden_dim) if use_plane_token else None
        vote_in_dim = hidden_dim * (2 if use_plane_token else 1)
        self.vote_head = VoteHead(vote_in_dim, hidden_dim)
        self.uncertainty_head = UncertaintyHead(hidden_dim) if use_uncertainty else None

    @staticmethod
    def _blend_to_logit(blend: float) -> float:
        blend = min(max(float(blend), 1e-4), 1.0 - 1e-4)
        return math.log(blend / (1.0 - blend))

    def set_refine_blend(self, blend: float) -> float:
        """Overwrite the final coarse-to-PCGV blend and return the clamped value."""
        blend = min(max(float(blend), 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            self.refine_blend_logit.fill_(self._blend_to_logit(blend))
        return blend

    def get_refine_blend(self) -> float:
        return float(torch.sigmoid(self.refine_blend_logit.detach()).cpu())

    def _identity(self, batch: int, device, dtype) -> torch.Tensor:
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)

    def _finite_or_previous(self, H_new: torch.Tensor,
                            H_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(H_new).flatten(1).all(dim=1).view(-1, 1, 1)
        fallback_count = (~finite.flatten()).sum().to(device=H_prev.device, dtype=H_prev.dtype)
        return torch.where(finite, H_new, H_prev), fallback_count

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor,
                H_init: Optional[torch.Tensor] = None,
                grid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Run PCGV from feature map A to feature map B.

        ``H_init`` and the returned ``H`` live in feature-grid pixel coordinates.
        """
        batch, channels, height, width = feat_a.shape
        if channels != self.feat_dim:
            raise ValueError(f"expected feat_dim={self.feat_dim}, got {channels}")

        if grid is None:
            grid = torch_make_pixel_grid(batch, height, width, feat_a.device, feat_a.dtype)
        else:
            grid = grid.to(device=feat_a.device, dtype=feat_a.dtype)

        H_t = H_init.to(device=feat_a.device, dtype=feat_a.dtype) if H_init is not None else self._identity(
            batch, feat_a.device, feat_a.dtype)
        H_t = torch_normalize_homography(H_t)
        H_start = H_t.clone()

        feat_a_n = F.normalize(feat_a, dim=1)
        feat_b_n = F.normalize(feat_b, dim=1)
        feat_a_tokens = sample_features_at_points(feat_a_n, grid, padding_mode="border")
        xy_norm = torch_pixel_to_norm(grid, height, width)
        h_state = feat_a.new_zeros(batch, grid.shape[1], self.hidden_dim)
        prev_vote = feat_a.new_full((batch, grid.shape[1], 1), 0.5)
        leverage = feat_a.new_zeros((batch, grid.shape[1], 1))

        stats = []
        residuals = None
        votes = None
        matches = None
        uncertainty = None
        solver_cond = feat_a.new_ones(batch)
        dlt_fallbacks = feat_a.new_zeros(())

        for _ in range(self.num_iters):
            center = torch_warp_points_h(grid, H_t)
            corr = correlation_match(
                feat_a_n,
                feat_b_n,
                center,
                radius=self.radius,
                grid_a=grid,
                temperature=self.temperature,
                padding_mode="zeros",
            )
            matches = corr["matches"]
            delta = corr["delta"]
            residuals = (matches - center).square().sum(dim=-1, keepdim=True).sqrt()
            feat_b_match = sample_features_at_points(feat_b_n, matches, padding_mode="zeros")

            scale = float(max(height - 1, width - 1, 1))
            entropy_scale = math.log(float((2 * self.radius + 1) ** 2))
            evidence = torch.cat(
                (
                    feat_a_tokens,
                    feat_b_match,
                    delta / scale,
                    residuals / scale,
                    corr["peak"],
                    corr["gap"],
                    corr["entropy"] / max(entropy_scale, 1e-6),
                    prev_vote,
                    leverage,
                    xy_norm,
                ),
                dim=-1,
            )
            z = self.encoder(evidence)
            h_state = self.vote_gru(z, h_state)
            if self.geometry_transformer is not None:
                h_state = self.geometry_transformer(h_state, xy_norm, delta / scale, residuals / scale)

            vote_input = h_state
            if self.plane_token is not None:
                vote_input = torch.cat((h_state, self.plane_token(h_state)), dim=-1)
            vote_logits = self.vote_head(vote_input)
            votes = self.min_vote + (1.0 - self.min_vote) * torch.sigmoid(vote_logits)

            H_new = torch_weighted_dlt(grid, matches, votes)
            H_new, fallback_count = self._finite_or_previous(H_new, H_t)
            dlt_fallbacks = dlt_fallbacks + fallback_count
            if self.damped_update:
                H_t = torch_normalize_homography(
                    self.update_alpha * H_new + (1.0 - self.update_alpha) * H_t
                )
            else:
                H_t = torch_normalize_homography(H_new)
            H_t, fallback_count = self._finite_or_previous(H_t, H_new)
            dlt_fallbacks = dlt_fallbacks + fallback_count
            prev_vote = votes

            if self.use_leverage:
                leverage = torch_dlt_leverage(grid, matches, votes).to(dtype=feat_a.dtype)
            else:
                leverage = feat_a.new_zeros((batch, grid.shape[1], 1))

            with torch.no_grad():
                try:
                    solver_cond = torch_dlt_condition_number(grid, matches, votes).to(dtype=feat_a.dtype)
                except RuntimeError:
                    solver_cond = feat_a.new_full((batch,), float("nan"))
            stats.append({
                "corr_peak": corr["peak"].detach(),
                "corr_gap": corr["gap"].detach(),
                "corr_entropy": corr["entropy"].detach(),
                "mean_vote": votes.detach().mean(dim=1),
                "mean_residual": residuals.detach().mean(dim=1),
            })

        H_refined = H_t
        refine_blend = torch.sigmoid(self.refine_blend_logit).to(device=feat_a.device, dtype=feat_a.dtype)
        H_t = torch_normalize_homography((1.0 - refine_blend) * H_start + refine_blend * H_refined)

        mask = votes.transpose(1, 2).reshape(batch, 1, height, width)
        if self.uncertainty_head is not None:
            uncertainty = self.uncertainty_head(h_state)
        else:
            uncertainty = residuals.detach() + 1e-4
        uncertainty_map = uncertainty.transpose(1, 2).reshape(batch, 1, height, width)

        return {
            "H": H_t,
            "H_refined": H_refined,
            "H_start": H_start,
            "mask": mask,
            "votes": votes,
            "matches": matches,
            "grid": grid,
            "residuals": residuals,
            "uncertainty": uncertainty_map,
            "uncertainty_tokens": uncertainty,
            "stats": stats,
            "solver_cond": solver_cond,
            "refine_blend": refine_blend.detach(),
            "dlt_fallbacks": dlt_fallbacks.detach(),
            "dlt_attempts": feat_a.new_tensor(float(batch * max(self.num_iters, 1) * 2)),
        }
