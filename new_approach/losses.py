"""Unsupervised losses for the no-mask baseline.

Two terms, both grounded in JirongZhang/DeepHomography (CA-UDHN):

* Alignment loss (triplet, mask == identity):
      L_align = mean( relu(1 + |Fb - Fa'| - |Fb - Fa|) )
  where Fb is the target feature, Fa the (un-warped) source feature and Fa'
  the source feature warped toward the target. The +1 margin and the negative
  |Fb - Fa| term are what stop the feature extractor from collapsing to a
  constant map (a plain L1 on the warped pair would not). Computed in both
  directions (forward I1<-I2 and backward I2<-I1).

* Feature Identity Loss (FIL):
      L_fil = mean( | warp(f(I)) - f(warp(I)) | )
  forces the feature extractor to commute with warping, which keeps the
  features geometrically meaningful. Also bidirectional.

HomoNet.forward already returns every tensor these need.
"""
import torch.nn.functional as F


def _l1(a, b):
    return (a - b).abs().mean()


def triplet_align(fb, fa_warp, fa, margin=1.0):
    """relu(margin + |Fb - Fa'| - |Fb - Fa|) averaged over all pixels."""
    return F.relu(margin + (fb - fa_warp).abs() - (fb - fa).abs()).mean()


def baseline_loss(out, lambda_align=1.0, lambda_fil=0.5, margin=1.0):
    """Total loss + a dict of scalar components for logging.

    `out` is the dict returned by HomoNet.forward.
    """
    f1 = out["img1_patch_fea"]          # Fb for the forward direction (target = I1)
    f2 = out["img2_patch_fea"]          # Fb for the backward direction (target = I2)
    f2_warp = out["warp_img2_patch_fea"]  # warp(f(I2)) -> aligned to I1   (Fa' fwd)
    f1_warp = out["warp_img1_patch_fea"]  # warp(f(I1)) -> aligned to I2   (Fa' bwd)
    f_warp2 = out["img2_patch_warp_fea"]  # f(warp(I2))                    (FIL fwd)
    f_warp1 = out["img1_patch_warp_fea"]  # f(warp(I1))                    (FIL bwd)

    align_f = triplet_align(f1, f2_warp, f2, margin)
    align_b = triplet_align(f2, f1_warp, f1, margin)
    fil_f = _l1(f2_warp, f_warp2)
    fil_b = _l1(f1_warp, f_warp1)

    align = align_f + align_b
    fil = fil_f + fil_b
    total = lambda_align * align + lambda_fil * fil

    return total, {
        "total": total.item(),
        "align": align.item(),
        "align_f": align_f.item(),
        "align_b": align_b.item(),
        "fil": fil.item(),
        "fil_f": fil_f.item(),
        "fil_b": fil_b.item(),
    }
