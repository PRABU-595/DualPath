import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    """
    Sets reproducibility seeds for random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 1. Deterministic algorithms for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 2. Environment variable for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"--- Reproducibility seeds set to {seed} ---")
