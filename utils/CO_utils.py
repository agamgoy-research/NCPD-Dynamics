import numpy as np
from math import *


def FCA(G, s, kappa, iteration):
    """Implements the Firefly Cellular Automata pulse coupled oscillator model
    Args:
        G (NetworkX Graph): Input graph to the model
        s (array): Current state
        k (int): k-color FCA
        iteration (int): Number of iterations
    Returns:
        ret (2D-np.array): States at each iteration
        label (bool): Whether the system concentrates at the final iteration
    """
    b = (kappa - 1) // 2  # Blinking color
    ret = s
    s_next = np.zeros(G.num_nodes())
    for h in range(iteration):
        if h != 0:
            s = s_next  # Update to the newest state
            ret = np.vstack((ret, s_next))
        s_next = np.zeros(G.num_nodes())
        for i in range(G.num_nodes()):
            flag = False  # True if inhibited by the blinking neighbor
            if s[i] > b:
                for j in range(G.num_nodes()):
                    if s[j] == b and list(G.nodes())[j] in list(G.neighbors(list(G.nodes())[i])):
                        flag = True
                if flag:
                    s_next[i] = s[i]
                else:
                    s_next[i] = (s[i] + 1) % kappa
            else:
                s_next[i] = (s[i] + 1) % kappa

    width = widthFCA(ret[-1], kappa)
    label = False

    if width < floor(kappa / 2):  # half circle concentration
        label = True

    return ret, label


def widthFCA(coloring, kappa=5):
    """Compute width of FCA dynamics coloring generated wiht a specific kappa
    Args:
        coloring (list): List of dynamics value at a particular iteration
        kappa (int, optional): Kappa value with which the FCA dynamics were generated. Defaults to 5.
    Returns:
        float: Width of the FCA dynamics
    """
    differences = [np.max(coloring) - np.min(coloring)]

    for j in range(1, kappa + 1):
        shifted = (np.array(coloring) + j) % kappa
        differences.append(np.max(shifted) - np.min(shifted))

    return np.min(differences)


# Kuramoto Dynamics
def Kuramoto(G, K, s, iteration, step=0.01):
    """Implements the Kuramoto model for coupled oscillators
    Args:
        G (NetworkX Graph): Input graph to the model
        K (float): Coupling strength of the Kuramoto dynamics
        s (list): Initial dynamic states of oscillators
        iteration (int): Number of iterations to run the dynamics for
    Returns:
        ret (2D-np.array): States at each iteration
        label (bool): Whether the system concentrates at the final iteration
    """
    ret = s
    s_next = np.zeros(G.num_nodes())
    for h in range(iteration - 1):
        if h != 0:
            s = s_next  # Update to the newest state
            ret = np.vstack((ret, s_next))
        for i in range(G.num_nodes()):
            neighbor_col = []
            for j in range(G.num_nodes()):
                if list(G.nodes())[j] in list(G.neighbors(list(G.nodes())[i])):
                    neighbor_col.append(s[j])

            new_col = s[i] + step * K * np.sum(np.sin(neighbor_col - s[i]))
            if np.abs(new_col) > np.pi:
                if new_col > np.pi:
                    new_col -= 2 * np.pi
                if new_col < -np.pi:
                    new_col += 2 * np.pi
            s_next[i] = new_col

    label = False
    if widthKURA(ret[-1]) < np.pi:
        label = True

    return ret, label


def widthKURA(colors):
    """Compute width of Kuramoto dynamics coloring
    Args:
        colors (list): List of dynamics value at a particular iteration
    Returns:
        float: Width of the Kuramoto dynamics
    """
    ordered = list(np.pi - np.asarray(colors))
    ordered.sort()
    lordered = len(ordered)
    threshold = np.pi

    if lordered == 1:
        return 0

    elif lordered == 2:
        dw = ordered[1] - ordered[0]
        if dw > threshold:
            return 2 * np.pi - dw
        else:
            return dw

    else:
        widths = [2 * np.pi + ordered[0] - ordered[-1]]
        for i in range(lordered - 1):
            widths.append(ordered[i + 1] - ordered[i])
        return np.abs(2 * np.pi - max(widths))
