"""Seed every source of randomness used during data splitting/training."""
from __future__ import annotations

import random


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass
