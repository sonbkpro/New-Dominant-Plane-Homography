import torch
import torch.nn.functional as F

try:
    from .modules.maskFlowHomo import make_flow_matching_batch
except ImportError:
    from modules.maskFlowHomo import make_flow_matching_batch

__all__ = [
    "bidirectional_mask_intersection",
    "feature_identity_loss",
    "flow_matching_loss",
    "mask_area_loss",
    "mask_entropy_loss",
    "mask_total_variation_loss",
    "masked_triplet_alignment_loss",
    "morphology_close_soft",
    "pseudo_mask_from_residuals",
]


def morphology_close_soft(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-dilated, kernel_size, stride=1, padding=pad)
    return eroded.clamp(0.0, 1.0)


def bidirectional_mask_intersection(
    img1_patch_mask: torch.Tensor,
    img2_patch_mask: torch.Tensor,
    warp_img1_patch_mask: torch.Tensor,
    warp_img2_patch_mask: torch.Tensor,
    use_close: bool = False,
    kernel_size: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_close:
        img1_patch_mask = morphology_close_soft(img1_patch_mask, kernel_size)
        img2_patch_mask = morphology_close_soft(img2_patch_mask, kernel_size)
        warp_img1_patch_mask = morphology_close_soft(warp_img1_patch_mask, kernel_size)
        warp_img2_patch_mask = morphology_close_soft(warp_img2_patch_mask, kernel_size)

    trusted_forward = img2_patch_mask * warp_img1_patch_mask
    trusted_backward = img1_patch_mask * warp_img2_patch_mask
    return trusted_forward, trusted_backward


def masked_triplet_alignment_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 1.0,
    eps: float = 1e-6,
    robust: str = "none",
    trunc_frac: float = 0.15,
    charbonnier_eps: float = 1e-3,
    tukey_c: float = 4.685,
) -> torch.Tensor:
    if mask.shape[-2:] != anchor.shape[-2:]:
        mask = F.interpolate(mask, size=anchor.shape[-2:], mode="bilinear", align_corners=False)
    if mask.shape[1] != 1:
        mask = mask.mean(dim=1, keepdim=True)

    ap = torch.abs(anchor - positive).mean(dim=1, keepdim=True)
    an = torch.abs(anchor - negative).mean(dim=1, keepdim=True)
    loss_map = F.relu(ap - an + float(margin))
    robust = str(robust).lower()
    if robust == "charbonnier":
        # Charbonnier is a mild smoother here, not a strong foreground outlier rejector.
        loss_map = torch.sqrt(loss_map.pow(2) + float(charbonnier_eps) ** 2) - float(charbonnier_eps)
    elif robust == "truncated":
        trunc_frac = min(max(float(trunc_frac), 0.0), 0.95)
        if trunc_frac > 0.0:
            flat = loss_map.flatten(1).float()
            cutoff = torch.quantile(flat, 1.0 - trunc_frac, dim=1, keepdim=True)
            cutoff = cutoff.to(dtype=loss_map.dtype).view(-1, 1, 1, 1)
            mask = mask * (loss_map <= cutoff).to(mask.dtype)
    elif robust == "tukey":
        c = max(float(tukey_c), eps)
        scaled = (loss_map / c).float()
        tukey_weight = torch.square(1.0 - scaled.square()).clamp_min(0.0)
        tukey_weight = torch.where(scaled < 1.0, tukey_weight, torch.zeros_like(tukey_weight))
        mask = mask * tukey_weight.to(mask.dtype)
    elif robust != "none":
        raise ValueError(f"unsupported robust alignment loss: {robust}")
    weighted = loss_map * mask
    denom = mask.flatten(1).sum(dim=1).clamp_min(eps)
    return (weighted.flatten(1).sum(dim=1) / denom).mean()


def feature_identity_loss(
    img1_patch_warp_fea: torch.Tensor,
    img2_patch_warp_fea: torch.Tensor,
    warp_img1_patch_fea: torch.Tensor,
    warp_img2_patch_fea: torch.Tensor,
) -> torch.Tensor:
    return F.l1_loss(img2_patch_warp_fea, warp_img2_patch_fea) + F.l1_loss(
        img1_patch_warp_fea, warp_img1_patch_fea
    )


def flow_matching_loss(
    mask_flow,
    cond: torch.Tensor,
    target_mask: torch.Tensor,
    noise_sigma: float = 1.0,
    t_min: float = 1e-4,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    z_t, t, target_velocity = make_flow_matching_batch(
        target_mask,
        noise_sigma=noise_sigma,
        t_min=t_min,
    )
    pred_velocity = mask_flow(z_t, t, cond)
    loss = (pred_velocity - target_velocity).pow(2)
    if confidence is not None:
        if confidence.shape[-2:] != loss.shape[-2:]:
            confidence = F.interpolate(confidence, size=loss.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss * confidence.clamp_min(0.0)
    return loss.mean()


def mask_area_loss(mask: torch.Tensor, min_area: float = 0.15, max_area: float = 0.9) -> torch.Tensor:
    area = mask.mean(dim=(1, 2, 3))
    low = F.relu(float(min_area) - area).pow(2)
    high = F.relu(area - float(max_area)).pow(2)
    return (low + high).mean()


def mask_total_variation_loss(mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[-1] < 2 or mask.shape[-2] < 2:
        return mask.new_tensor(0.0)
    loss_h = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
    loss_w = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
    return loss_h + loss_w


def mask_entropy_loss(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.clamp(eps, 1.0 - eps)
    entropy = -(mask * torch.log(mask) + (1.0 - mask) * torch.log(1.0 - mask))
    return entropy.mean()


def _robust_normalize(residual: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    batch_size = residual.shape[0]
    flat = residual.flatten(1)
    median = flat.median(dim=1).values.view(batch_size, 1, 1, 1)
    mad = torch.abs(residual - median).flatten(1).median(dim=1).values.view(batch_size, 1, 1, 1)
    return (residual - median) / (mad + eps)


def pseudo_mask_from_residuals(
    reference_feature: torch.Tensor,
    warped_feature: torch.Tensor,
    reference_image: torch.Tensor | None = None,
    warped_image: torch.Tensor | None = None,
    tau: float = 1.0,
    temperature: float = 0.5,
    close_kernel: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if warped_feature.shape[-2:] != reference_feature.shape[-2:]:
        warped_feature = F.interpolate(
            warped_feature,
            size=reference_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    residual = torch.abs(reference_feature - warped_feature).mean(dim=1, keepdim=True)
    residual = _robust_normalize(residual)

    if reference_image is not None and warped_image is not None:
        if warped_image.shape[-2:] != reference_feature.shape[-2:]:
            warped_image = F.interpolate(
                warped_image,
                size=reference_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if reference_image.shape[-2:] != reference_feature.shape[-2:]:
            reference_image = F.interpolate(
                reference_image,
                size=reference_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        image_residual = torch.abs(reference_image - warped_image).mean(dim=1, keepdim=True)
        image_residual = _robust_normalize(image_residual)
        residual = 0.7 * residual + 0.3 * image_residual

    mask = torch.sigmoid((float(tau) - residual) / max(float(temperature), 1e-6))
    if close_kernel > 1:
        mask = morphology_close_soft(mask, close_kernel)
    confidence = (mask - 0.5).abs() * 2.0
    return mask.clamp(0.0, 1.0), confidence.clamp(0.0, 1.0)
