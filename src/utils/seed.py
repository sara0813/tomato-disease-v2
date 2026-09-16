"""재현성을 위한 시드 고정."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """random / numpy / torch 시드를 한 번에 고정한다."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # GPU가 없어도 무해함
