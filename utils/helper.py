from random import random
import numpy as np


def set_seed(seed):
    """Sets seed for reproducibility
    Args:
        seed (int): Reproducibility seed to use for the experiments
    """
    np.random.seed(seed)


def args_path(args, num_nodes, sample_size):
    """Determines save path for data tensors

    Args:
        args (argparse): Python argparse arguments entity

    Returns:
        str: Output path for saving data
    """
    return f"Model_{args.model}_Network_{args.network}_SampleK_{str(args.samplek)}_ParentNodes_{num_nodes}_NumSamples_{sample_size}_Seed_{args.seed}"
