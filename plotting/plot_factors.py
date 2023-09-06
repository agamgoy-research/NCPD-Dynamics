import os
import sys
import pickle
import argparse
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings

sys.path.insert(0, "/home/agoyal25/NCPD-Dynamics")
from utils import *

warnings.filterwarnings("ignore")

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


def plot_CP(temporal_factors, graph_factors, R, retrieve_path):
    fig = plt.figure(figsize=(15, R // 4 * 3.5))
    rows = R // 4
    columns = 4
    gs = gridspec.GridSpec(rows, columns, wspace=0.05, hspace=0.05)

    height_ratios = [1, 9]

    for i in range(rows):
        for j in range(columns):
            idx = 3 * i + (i + j)

            histogram = temporal_factors[:, idx].reshape(-1, 1).T
            k = int(np.sqrt(graph_factors[:, idx].shape[0]))
            G_arr = graph_factors[:, idx].reshape(k, k)
            G = nx.from_numpy_array(G_arr)
            edges = G.edges()
            weights = [1 * G[u][v]["weight"] for u, v in edges]

            sub_gs = gs[i, j].subgridspec(2, 1, height_ratios=height_ratios)

            ax1 = plt.subplot(sub_gs[0])
            sns.heatmap(
                histogram,
                cmap="viridis",
                cbar=False,
                vmin=0,
                vmax=1,
                xticklabels=False,
                yticklabels=False,
                ax=ax1,
            )
            ax1.set_xticks([])
            ax1.set_yticks([])

            ax2 = plt.subplot(sub_gs[1])
            nx.draw_spring(G, with_labels=False, node_size=20, width=weights, label="Graph", ax=ax2)
            ax2.set_xticks([])
            ax2.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.show()

    plt.savefig(os.path.join(retrieve_path, f"factors_rank{R}.jpeg"))


ranks = [temporal_factors[j].shape[1] for j in range(len(temporal_factors))]

for idx, R in enumerate(ranks):
    plot_CP(temporal_factors[idx], graph_factors[idx], R, retrieve_path)
