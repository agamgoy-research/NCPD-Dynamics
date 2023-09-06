import os
import pickle
import argparse
import numpy as np
from utils import *
import tensorly as tl
from tensorly.decomposition import non_negative_parafac

# Build argument parser
parser = argparse.ArgumentParser(description="Argument Parser for Nonnegative CP Decomposition")
parser.add_argument(
    "-m",
    "--model",
    default="FCA",
    type=str,
    choices=["FCA", "KURA", "HK"],
    help="Dynamics model to use for decomposition",
)
parser.add_argument(
    "-n",
    "--network",
    default="NWS",
    type=str,
    choices=["NWS", "BA", "ER"],
    help="Underlying Parent Network that subgraphs were sampled from",
)
parser.add_argument(
    "-k",
    "--samplek",
    default=15,
    type=int,
    help="Number of nodes in the sampled subgraphs",
)
parser.add_argument(
    "-data",
    "--data_dir",
    default="data",
    type=str,
    help="Directory to retrieve data from",
)
parser.add_argument(
    "-seed",
    "--seed",
    default=0,
    type=int,
    help="Set reproducibility seed",
)
parser.add_argument(
    "-r", "--rank_list", nargs="+", type=int, help="List of CP-Decomposition Ranks", required=True
)
args = parser.parse_args()

# Load tensor data
num_nodes, sample_size = 450, 2500
key = args_path(args, num_nodes, sample_size)
tensor = tl.tensor(
    np.load(
        os.path.join(
            args.data_dir,
            key,
            key + ".npy",
        )
    )
)
print(f"Shape of created tensor is {tensor.shape}.\n")

dynamics_iter = tensor.shape[1]
ranks = args.rank_list

# Run NCPD on data tensor
NCPD_factors = [
    non_negative_parafac(
        tensor,
        rank=R,
        n_iter_max=1000,
        tol=1e-6,
        return_errors=True,
        verbose=1,
        random_state=args.seed,
    )
    for R in ranks
]
all_factors = [NCPD_factors[i] for i in range(len(NCPD_factors))]
reconstruction_list = [all_factors[j][1] for j in range(len(all_factors))]

# Shape: (dynamics_iter, R)
temporal_factors = [all_factors[j][0].factors[1] for j in range(len(all_factors))]
for i in range(len(temporal_factors)):
    scales = np.linalg.norm(temporal_factors[i], ord=np.inf, axis=0)  # Normalize the factors
    temporal_factors[i] /= scales

# Shape: (samplek**2, R)
graph_factors = [all_factors[i][0].factors[2] for i in range(len(all_factors))]
for i in range(len(graph_factors)):
    scales = np.linalg.norm(graph_factors[i], ord=np.inf, axis=0)  # Normalize the factors
    graph_factors[i] /= scales

## Need to redirect to plotting scripts
save_path = os.path.join(args.data_dir, key)
with open(os.path.join(key, "graph_factors.pkl"), "wb") as f:
    pickle.dump(graph_factors, f)
with open(os.path.join(key, "temporal_factors.pkl"), "wb") as f:
    pickle.dump(temporal_factors, f)
