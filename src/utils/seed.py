"""재현성을 위한 시드 고정."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """random / numpy / tensorflow 시드를 한 번에 고정한다."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)
