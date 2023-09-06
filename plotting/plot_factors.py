import os
import sys
import pickle
import argparse
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

sys.path.insert(0, "/home/agoyal25/NCPD-Dynamics")
from utils import *

# Build argument parser
parser = argparse.ArgumentParser(description="Argument Parser for Plotting Scripts")
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
args = parser.parse_args()

num_nodes, sample_size = 450, 2500
key = args_path(args, num_nodes, sample_size)
retrieve_path = os.path.join(args.data_dir, key)

with open(os.path.join(retrieve_path, "graph_factors.pkl"), "rb") as f:
    graph_factors = pickle.load(f)

with open(os.path.join(retrieve_path, "temporal_factors.pkl"), "rb") as f:
    temporal_factors = pickle.load(f)

print("Loaded data successfully...")
