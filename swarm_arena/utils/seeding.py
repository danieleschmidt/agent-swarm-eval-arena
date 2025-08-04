"""Seeding utilities for reproducible experiments."""

import random
import numpy as np
from typing import Optional


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set additional seeds for other libraries if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import ray
        # Ray uses its own seeding mechanism
        pass
    except ImportError:
        pass


def generate_seed() -> int:
    """Generate a random seed.
    
    Returns:
        Random integer seed
    """
    return np.random.randint(0, 2**31 - 1)