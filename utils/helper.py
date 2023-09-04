from random import random
import numpy as np


def set_seed(seed):
    """Sets seed for reproducibility
    Args:
        seed (int): Reproducibility seed to use for the experiments
    """
    np.random.seed(seed)
    random.seed(seed)


def args_path(args):
    """Determines save path for data tensors

    Args:
        args (argparse): Python argparse arguments entity

    Returns:
        str: Output path for saving data
    """
    return f"Model_{args.model}_Network_{args.network}_SampleK_{str(args.samplek)}_Seed_{args.seed}"
