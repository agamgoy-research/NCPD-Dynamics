import os
import sys
import copy
import argparse
import networkx as nx
from NNetwork import NNetwork as nn

sys.path.insert(0, "/home/agoyal25/NCPD-Dynamics")
from utils import *

# Build argument parser
parser = argparse.ArgumentParser(
    description="Argument Parser for Coupled Oscillator Dynamics Tensor Generation"
)
parser.add_argument(
    "-m",
    "--model",
    default="FCA",
    type=str,
    choices=["FCA", "KURA"],
    help="Coupled Oscillator Dynamics model to use for simulation",
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
parser.add_argument(
    "-seed",
    "--seed",
    default=0,
    type=int,
    help="Set reproducibility seed",
)
args = parser.parse_args()

# Set seed
set_seed(args.seed)

# Lookup table for Dynamics and Width
widthTable = {"FCA": widthFCA, "KURA": widthKURA}
phiTable = {"FCA": FCA, "KURA": Kuramoto}

# Dynamics Model and Graph Statistics
num_nodes, probability, auxiliary, sample_size = 300, 0.25, 10, 2500
sampling_alg = "pivot"

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

# Create main tensor
final_tensor = []
for row in X:
    A_new = row.reshape(args.samplek, args.samplek)
    G_new = nx.from_numpy_array(A_new)

    new_nodes = {e: n for n, e in enumerate(G_new.nodes, start=1)}
    new_edges = [(new_nodes[e1], new_nodes[e2]) for e1, e2 in G_new.edges]
    edgelist = []
    for i in range(len(new_edges)):
        temp = [str(new_edges[i][0]), str(new_edges[i][1])]
        edgelist.append(temp)
    G = nn.NNetwork()
    G.add_edges(edgelist)
    num_nodes = G.num_nodes()

    if args.model == "FCA":
        s = np.random.randint(0, 5, num_nodes)
        kappa = 5
        dynamics, label = phiTable[args.model](G, s, kappa, iteration=50)
    elif args.model == "KURA":
        s = np.random.uniform(-np.pi, np.pi, num_nodes)
        dynamics, label = phiTable[args.model](G, s=s, K=1, iteration=100)
    else:
        raise NotImplementedError(f"{args.model} is not yet supported.")

    # Create individual CCATs
    tensor = []
    for color in dynamics:
        adj_mat = copy.deepcopy(A_new)

        for j in range(args.samplek - 1):
            for k in range(j):
                if adj_mat[j, k] > 0:
                    adj_mat[j, k] = widthTable[args.model]([color[j], color[k]])
        adj_mat += adj_mat.T
        tensor.append(
            adj_mat.reshape(
                args.samplek**2,
            )
        )
    final_tensor.append(np.array(tensor))

final_tensor = np.array(final_tensor)

if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)
np.save(os.path.join(args.data_dir, args_path(args, num_nodes, sample_size)), final_tensor)
