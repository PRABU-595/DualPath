import torch

def get_device():
    """
    Automatically detects the best available hardware for training/inference.
    Prioritizes CUDA, then MPS (Apple Silicon), then CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_autocast_device_type():
    """
    Returns the device type string for torch.autocast.
    """
    device = get_device()
    if device == "cuda":
        return "cuda"
    # mps and cpu use "cpu" for autocast or don't support it in the same way
    return "cpu"

def get_dtype(device: str):
    """
    Returns the appropriate floating point precision based on device.
    """
    if device == "cuda":
        # bf16 is best for Ampere/L4/A100, float16 is safer for older GPUs
        return torch.bfloat16
    elif device == "mps":
        return torch.float16
    else:
        return torch.float32
