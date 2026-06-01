"""One-batch diagnostic: is the catastrophic eval PME a BatchNorm train/eval mismatch?

Run on the machine that has the checkpoint:
    python new_approach/_diag_bn.py
Edit CKPT / DEV below if your paths differ.

Interpretation:
  If 'eval (BN running stats)' shows a HUGE flow_f / H magnitude and
  'eval + BN batch-stats' shows a SMALL one  -> BatchNorm is the cause.
Delete this file afterwards.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data.homo_flow_dataset import build_homo_flow_loader
from new_approach.geometry_homo import flow_mask_to_homography
from new_approach.modules.transformerHomo import Ms_Transformer

CKPT = "new_approach/checkpoints/_probe5k/pretrain_homo_final.pth"
DEV = "cuda:3"
TEST_LIST = "Data/Test_List.txt"
TEST_IMAGE_DIR = "Data/Test"
COORD_DIR = "Data/Coordinate-v2/Coordinate-v2"

device = torch.device(DEV if torch.cuda.is_available() else "cpu")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
params = SimpleNamespace(**ck["params"])
params.return_h_matrix = False
# defensive backfill in case an older checkpoint predates these fields
for name, val in {"refine_iters": 1, "feature_channels": 1, "use_correlation": False,
                  "corr_radius": 4, "est_ref_channels": 1}.items():
    if not hasattr(params, name):
        setattr(params, name, val)

model = Ms_Transformer(params=params).to(device)
state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"loaded {CKPT}  missing={len(missing)} unexpected={len(unexpected)}")

loader = build_homo_flow_loader(
    repo_root=ROOT, list_path=TEST_LIST, image_dir=TEST_IMAGE_DIR,
    crop_size=tuple(params.crop_size), full_size=(360, 640), rho=16, shift=8,
    batch_size=8, shuffle=False, num_workers=2, training=False,
    horizontal_flip_aug=False, coordinate_dir=COORD_DIR, max_items=8, seed=230,
)
batch = next(iter(loader))
batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def report(tag):
    with torch.no_grad():
        out = model(batch)
    f = out["flow_f"]
    H = flow_mask_to_homography(f, None)
    print(f"{tag:30s} | flow_f abs: max={f.abs().max().item():10.2f} mean={f.abs().mean().item():8.3f} "
          f"| H abs max={H.abs().max().item():10.2f}")


model.eval()
report("eval (BN running stats)")

# force only BatchNorm layers to use batch statistics (keeps the correct eval forward path)
n_bn = 0
for m in model.modules():
    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        m.train()
        n_bn += 1
report(f"eval + BN batch-stats ({n_bn} BN)")
