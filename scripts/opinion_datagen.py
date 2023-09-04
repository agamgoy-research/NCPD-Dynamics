import sys
import argparse
import networkx as nx
from NNetwork import NNetwork as nn

sys.path.insert(0, "/home/agoyal25/NCPD-Dynamics")
from utils import *

# Build argument parser
parser = argparse.ArgumentParser(description="Argument Parser for Opion Dynamics Tensor Generation")
parser.add_argument(
    "-N",
    "--N",
    default=500,
    type=int,
    help="Number of parents from which 50 subgraphs each were sampled",
)
parser.add_argument(
    "-m",
    "--model",
    default="HK",
    type=str,
    choices=["HK"],
    help="Opinion Dynamics model to use for simulation",
)
parser.add_argument(
    "-n",
    "--network",
    default="NWS",
    type=str,
    choices=["NWS", "BA", "ER"],
    help="Underlying Parent Network to sample subgraphs from",
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
    help="Directory to store generated data",
)
args = parser.parse_args()

# Lookup table for Dynamics and Width
widthTable = {"HK": widthOD}

# Dynamics Model and Graph Statistics
num_nodes, probability, auxiliary, sample_size = 450, 0.25, 10, 2500
sampling_alg = "pivot"

subgraphs = []

# Large graph generation
if args.network == "NWS":
    G_net = nx.newman_watts_strogatz_graph(num_nodes, auxiliary, probability)
elif args.network == "BA":
    G_net = nx.barabasi_albert_graph(num_nodes, auxiliary)
elif args.network == "ER":
    G_net = nx.erdos_renyi_graph(num_nodes, probability)
else:
    raise NotImplementedError(f"{args.network} is not yet supported.")

A = nx.adjacency_matrix(G_net)
new_nodes = {e: n for n, e in enumerate(G_net.nodes, start=1)}
new_edges = [(new_nodes[e1], new_nodes[e2]) for e1, e2 in G_net.edges]
edgelist = []
for i in range(len(new_edges)):
    temp = [str(new_edges[i][0]), str(new_edges[i][1])]
    edgelist.append(temp)
G_nn = nn.NNetwork()
G_nn.add_edges(edgelist)

X, embs = G_nn.get_patches(k=args.samplek, sample_size=sample_size, skip_folded_hom=True)
X = X.T
print("Finished MCMC Sampling...")
