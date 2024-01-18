import numpy as np
from run_functions import *
from agent_functions import *
from routing_networks import large_braess_network
import math
import igraph as ig
from recommenders import general_heuristic_recommender


def single_run(n_agents, n_states, n_actions, n_iter, epsilon, gamma, alpha, q_initial, recommender):

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
    weights = ["x", 1,
               "x", 1, 0,
               "x", 1,
               0, 1,
               "x", 1, 0,
               "x",
               1, 0,
               "x"]
    g.es["cost"] = weights
    adj = g.get_adjacency(attribute="cost")
    paths = g.get_all_simple_paths(0, to=8)

    Q = initialize_q_table(q_initial, n_agents, n_states, n_actions)
    alpha = initialize_learning_rates(alpha, n_agents)
    S = np.random.randint(0, n_states, size=n_agents)

    eps_decay = n_iter/8
    if epsilon == "DECAYED":
        eps_start = 1
        eps_end = 0
    else:
        eps_start = epsilon
        eps_end = epsilon

    data = {}
    ind = np.arange(n_agents)

    for t in range(n_iter):
        epsilon = (eps_end + (eps_start - eps_end) * math.exp(-1. * t / eps_decay))

        if recommender == "optimized":
            S = general_heuristic_recommender(Q, S, n_agents, n_states, n_actions)
        elif recommender == "random":
            S = np.random.randint(0, n_states, size=n_agents)
        elif recommender == "constant":
            pass

        A = e_greedy_select_action(Q, S, epsilon)
        R = large_braess_network(A, paths, adj, n_agents)
        Q, sum_of_belief_updates = bellman_update_q_table(Q, S, A, R, alpha, gamma)

        ## SAVE PROGRESS DATA
        data[t] = {
                   # "nA": np.bincount(A, minlength=3),
                   "R": R,
                   # "Qmean": Q.mean(axis=1).mean(axis=0),
                   # "groups": count_groups(Q[ind, S, :], 0.1),
                   "Qvar": Q[ind, S, :].var(axis=0),
                   # "T": travel_time_per_route,
                   # "sum_of_belief_updates": sum_of_belief_updates,
                   # "alignment": calculate_alignment(Q, S, A),
                   # "S": S,
                   # "A": A,
                   # "Q": Q,
                   }
    return data


if __name__ == "__main__":
    from routing_networks import braess_augmented_network, braess_initial_network
    from recommenders import heuristic_recommender, constant_recommender
    from plot_functions import plot_run

    N_AGENTS = 100
    N_STATES = 4
    N_ACTIONS = 3
    N_ITER = 10000

    EPSILON = 0.01
    GAMMA = 0
    ALPHA = 0.01
    QINIT = "UNIFORM"  # np.array([-2, -2, -2])

    M = single_run(N_AGENTS, N_STATES, N_ACTIONS, N_ITER, EPSILON, GAMMA, ALPHA, QINIT, "constant")

    NAME = f"run_N{N_AGENTS}_S{N_STATES}_A{N_ACTIONS}_I{N_ITER}_e{EPSILON}_g{GAMMA}_a{ALPHA}_q{QINIT}"

    # plot_run(M, NAME, N_AGENTS, N_ACTIONS, N_ITER)
