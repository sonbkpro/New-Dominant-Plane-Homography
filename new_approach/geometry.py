"""Convert the network's crop-local homography flow into a 3x3 homography.

HomoGAN's transformer regresses a dense displacement field on the cropped patch.
During training that patch flow is applied to full-frame coordinates by adding
the crop origin (`start`). Therefore the correct 3x3 evaluation homography is
recovered from the four crop corners in full-image coordinates, not from the
outer full-frame corners.
"""
import cv2
import numpy as np


def flow_to_homography(flow, start_xy=(0.0, 0.0)):
    """Build H_ab from crop flow.

    Args:
        flow: np.ndarray with shape (crop_h, crop_w, 2), displacement in pixels.
        start_xy: (x, y) origin of this crop in the full image.

    Returns:
        3x3 homography mapping image-1 full-frame pixels to image-2 pixels.
    """
    crop_h, crop_w = flow.shape[:2]
    start_x, start_y = map(float, start_xy)
    local = np.float32([[0, 0],
                        [crop_w - 1, 0],
                        [crop_w - 1, crop_h - 1],
                        [0, crop_h - 1]])
    src = local + np.float32([start_x, start_y])
    dst = np.empty_like(src)
    for i, (x, y) in enumerate(local):
        fx = flow[int(y), int(x), 0]
        fy = flow[int(y), int(x), 1]
        dst[i] = src[i] + np.float32([fx, fy])
    return cv2.getPerspectiveTransform(src, dst)


def geometric_distance(p1, p2, H):
    """L2 between H * p1 (homogeneous, de-homogenised) and p2."""
    v = np.array([p1[0], p1[1], 1.0])
    est = H @ v
    est = est / est[2]
    return float(np.hypot(est[0] - p2[0], est[1] - p2[1]))
