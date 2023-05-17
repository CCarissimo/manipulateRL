import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from run_functions import *
from agent_functions import *
from plot_functions import *
from routing_networks import braess_initial_network
import tqdm

N_AGENTS = 100
N_STATES = 2
N_ACTIONS = 2
N_ITER = 500

EPSILON = 0.2
mask = np.zeros(N_AGENTS)
mask[:] = 1
GAMMA = 0
ALPHA = 0.01
QINIT = np.array([-1.5, -1.5])


def constant_recommender(t):
    return np.concatenate([np.zeros(int(N_AGENTS / 2)).astype(int), np.ones(int(N_AGENTS / 2)).astype(int)])


def random_recommender(t):
    return np.random.randint(0, 2, size=N_AGENTS)


def aligned_recommender(t):
    if t > 0:
        return np.concatenate([np.zeros(int(N_AGENTS / 2)).astype(int), np.ones(int(N_AGENTS / 2)).astype(int)])
    else:
        return np.ones(N_AGENTS).astype(int)


def misaligned_recommender(t):
    if t > 0:
        return np.concatenate([np.zeros(int(N_AGENTS / 2)).astype(int), np.ones(int(N_AGENTS / 2)).astype(int)])
    else:
        return np.zeros(N_AGENTS).astype(int)


def single_run(ALPHA, EPSILON, QINIT, N_AGENTS, N_STATES, N_ACTIONS, mask, recommender):

    QINIT = np.array([-1.5, -1.5])

    Q = initialize_q_table(QINIT, N_AGENTS, N_STATES, N_ACTIONS)

    if ALPHA == "UNIFORM":
        ALPHA = np.random.random_sample(size=N_AGENTS)

    if EPSILON == "UNIFORM":
        EPSILON = np.random.random_sample(size=N_AGENTS) * mask
    else:
        EPSILON = EPSILON * np.ones(N_AGENTS) * mask

    M = {}
    ind = np.arange(N_AGENTS)

    for t in range(N_ITER):
        S = recommender(t)

        A = e_greedy_select_action(Q, S, EPSILON)

        R, travel_time_per_route = braess_initial_network(A)

        Q, sum_of_belief_updates = bellman_update_q_table(Q, S, A, R, ALPHA, GAMMA)

        alignment, recommendation_alignment, action_alignment = calculate_alignment(Q, S, A)

        ## SAVE PROGRESS DATA
        M[t] = {"nA": np.bincount(A, minlength=3),
                "R": R,
                "Qmean": Q.mean(axis=1).mean(axis=0),
                "groups": count_groups(Q[ind, S, :], 0.1),
                "Qvar": Q[ind, S, :].var(axis=0),
                "T": travel_time_per_route,
                "sum_of_belief_updates": sum_of_belief_updates,
                "alignment": alignment,
                "recommendation_alignment": recommendation_alignment,
                "action_alignment": action_alignment,
                }
    return M


if __name__ == "__main__":
    NAME = "two_route_experiments_2"
    results = []
    for epsilon in tqdm.tqdm(np.linspace(0, 1, 51)):
        for recommender in [constant_recommender, random_recommender, aligned_recommender, misaligned_recommender]:
            for repetitions in range(10):
                M = single_run(ALPHA, epsilon, QINIT, N_AGENTS, N_STATES, N_ACTIONS, mask, recommender)

                W = [M[t]["R"].mean() for t in range(0, N_ITER)]
                T = np.mean(W[int(0.8 * N_ITER):N_ITER])
                T_all = np.mean(W)
                T_std = np.std(W[int(0.8 * N_ITER):N_ITER])

                Qvar = [M[t]["Qvar"] for t in range(0, N_ITER)]
                Qvar_mean = np.mean(Qvar)

                alignment = np.array([M[t]["recommendation_alignment"] for t in range(int(0.8 * N_ITER), N_ITER)])
                alignment_all = np.array([M[t]["recommendation_alignment"] for t in range(N_ITER)]).mean(axis=0)
                alignment = alignment.mean(axis=0)

                row = {
                    "epsilon": epsilon,
                    "T_mean": T,
                    "T_mean_all": T_all,
                    "T_std": T_std,
                    "repetition": repetitions,
                    "Qvar_mean": Qvar_mean,
                    "recommender_type": recommender.__name__,
                    "alignment": alignment,
                    "alignment_all": alignment_all
                }

                results.append(row)

    df = pd.DataFrame(results)

    df.to_csv(NAME + ".csv")
