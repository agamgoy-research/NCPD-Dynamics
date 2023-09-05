import numpy as np
from numpy.linalg import norm
from tqdm import trange


def preprocessArgs(s, max_rounds):
    """Argument processing function
    Source: https://github.com/dmpalyvos/opinions-research
    Args:
        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector
        max_rounds (int): Maximum number of rounds to simulate
    Returns:
        N (int): Length of opinion vector (number of agents to simulate)
        z (1xN numpy array): Copy of initial opinions (intrinsic beliefs) vector
        max_rounds (int): Updated maximum number of rounds to simulate
    """
    N = np.size(s)
    max_rounds = int(max_rounds) + 1  # Round 0 contains the initial opinions
    z = s.copy()

    return N, z, max_rounds


def hk_local(A, s, op_eps, max_rounds, eps=1e-6, conv_stop=True):
    """Simulates the model of Hegselmann-Krause with an Adjacency Matrix.
    Contrary to the standard Hegselmann-Krause Model, here we make use of
    an adjacency matrix that represents an underlying social structure
    independent of the opinions held by the members of the society.
    Source: https://github.com/dmpalyvos/opinions-research
    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)
        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector
        op_eps: ε parameter of the model
        max_rounds (int): Maximum number of rounds to simulate
        eps (double): Maximum difference between rounds before we assume that the model has converged. Defaults to 1e-6.
        conv_stop (bool): Stop the simulation if the model has converged. Defaults to True.
    Returns:
        txN numpy array: Vector of the opinions of the nodes over time

    """
    N, z, max_rounds = preprocessArgs(s, max_rounds)

    # All nodes must listen to themselves for the averaging to work
    A_model = A + np.eye(N)

    # The matrix contains 0/1 values
    A_model = A_model.astype(np.int8)

    z_prev = z.copy()
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in trange(1, max_rounds):
        for i in range(N):
            # Neighbors in the underlying social network
            neighbor_i = A_model[i, :] > 0
            opinion_close = np.abs(z_prev - z_prev[i]) <= op_eps
            # The node listens to those who share a connection with him in the underlying
            # network and also have an opinion which is close to his own
            friends_i = np.logical_and(neighbor_i, opinion_close)
            z[i] = np.mean(z_prev[friends_i])
        opinions[t, :] = z
        z_prev = z.copy()
        if conv_stop and norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print("Hegselmann-Krause (Local Knowledge) converged after {t} " "rounds".format(t=t))
            break

    return opinions[0 : t + 1, :]


def widthOD(colors):
    """Calculate the width of the opinion dynamics profile
    Args:
        colors (list): List of opinion dynamics vector profile at a certain iteration
    Returns:
        float: Width of the Hegselmann-Krause opinion dynamics
    """
    return np.abs(np.max(colors) - np.min(colors))
