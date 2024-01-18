import numpy as np
import igraph as ig


g = ig.Graph(
        9,
        [(0, 1), (0, 2),
         (1, 3), (1, 4), (1, 2),
         (2, 4), (2, 5),
         (3, 4), (3, 6),
         (4, 6), (4, 7), (4, 5),
         (5, 7),
         (6, 8), (6, 7),
         (7, 8)],
        directed=True
    )
weights = ["x", 1, "x", 1, 0, "x", 1, 0, 1, "x", 1, 0, "x", 1, 0, "x"]
g.es["cost"] = weights
adj = g.get_adjacency(attribute="cost")
paths = g.get_all_simple_paths(0, to=8)


def large_braess_network(A, paths, adj, n_agents):
    visits = np.zeros((9, 9))
    for a in A:
        path = paths[a]
        for i in range(len(path) - 1):
            node = path[i]
            next_node = path[i + 1]
            visits[node, next_node] += 1

    costs = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            if adj[i, j] == 'x':
                costs[i, j] = visits[i, j] / 100
            elif adj[i, j] == 1:
                costs[i, j] = 1

    R = np.zeros((n_agents))
    for agent, a in enumerate(A):
        path = paths[a]
        cost = 0
        for i in range(len(path) - 1):
            node = path[i]
            next_node = path[i + 1]
            cost += costs[node, next_node]
        R[agent] = cost
    return -R


def braess_augmented_network(A):
    n_agents = len(A)
    n_up = (A == 0).sum()
    n_down = (A == 1).sum()
    n_cross = (A == 2).sum()

    r_0 = 1 + (n_up + n_cross) / n_agents
    r_1 = 1 + (n_down + n_cross) / n_agents
    r_2 = (n_up + n_cross) / n_agents + (n_down + n_cross) / n_agents
    T = [-r_0, -r_1, -r_2]

    R = np.array([T[a] for a in A])  # -1 * np.vectorize(dict_map.get)(A)
    return R, T


def braess_initial_network(A):
    n_agents = len(A)
    n_up = (A == 0).sum()
    n_down = (A == 1).sum()

    r_0 = 1 + n_up / n_agents
    r_1 = 1 + n_down / n_agents
    T = [-r_0, -r_1]

    R = np.array([T[a] for a in A])
    return R, T


def two_route_game(A):
    n_agents = len(A)
    n_up = (A == 0).sum()

    r_0 = n_up / n_agents
    r_1 = 1
    T = [-r_0, -r_1]

    R = np.array([T[i] for i in A])
    return R, T


def minority_game(A, threshold=0.4):
    n_agents = len(A)
    n_up = (A == 0).sum()
    
    if n_up/n_agents < threshold:  # up is minority
        r_0 = 1
        r_1 = 0
        s = 0
    elif (n_agents - n_up)/n_agents < threshold:  # down is minority
        r_0 = 0
        r_1 = 1
        s = 1
    else:
        r_0 = 0
        r_1 = 0
        s = 2
    
    T = [r_0, r_1]

    R = np.array([T[i] for i in A])
    return R, T, np.broadcast_to(s, n_agents)
