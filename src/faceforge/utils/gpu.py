"""GPU and device utilities."""
import torch

def select_device(preferred: str = "cpu") -> str:
    if preferred.startswith("cuda") or preferred == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "cpu"
