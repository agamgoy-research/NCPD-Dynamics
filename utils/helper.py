from random import random
import numpy as np


def set_seed(seed):
    """Sets seed for reproducibility

    Args:
        seed (int): Reproducibility seed to use for the experiments
    """
    np.random.seed(seed)
    random.seed(seed)
