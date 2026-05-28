import warnings

import torch


def cuda_is_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        probe = torch.ones(1, device="cuda")
        probe = probe + 1
        torch.cuda.synchronize()
        return bool(probe.item() == 2.0)
    except Exception as exc:
        warnings.warn(
            "CUDA is visible but unusable by this PyTorch build. "
            f"Falling back to CPU. Original CUDA error: {exc}",
            RuntimeWarning,
        )
        return False


def resolve_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if cuda_is_usable() else "cpu")
    if requested.startswith("cuda") and not cuda_is_usable():
        warnings.warn(f"Requested {requested}, but CUDA is unusable. Using CPU instead.", RuntimeWarning)
        return torch.device("cpu")
    return torch.device(requested)
