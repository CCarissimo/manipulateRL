
def main(epsilon):  # pass epsilon as an argument using argparse
    import numpy as np
    import tqdm
    import pickle
    import nolds
    import pandas as pd
    from recommenders import random_recommender, constant_recommender, optimized_heuristic_recommender, aligned_heuristic_recommender
    from single_run import single_run
    from routing_networks import braess_augmented_network
    import os

    # Base Settings Which Will Not Change
    N_AGENTS = 100
    N_STATES = 3
    N_ACTIONS = 3
    N_ITER = 100  #10000
    N_REPEATS = 1
    mask = np.zeros(N_AGENTS)
    mask[:] = 1
    GAMMA = 0
    ALPHA = 0.1

    # Parameters which will be Varied
    epsilons = [epsilon]
    
    QINIT = "Variable"
    sizeQinit = 4
    qinits = {
        "uniform": "UNIFORM",
        "nash": np.array([-2, -2, -2]),
        "aligned": "ALIGNED",
        "misaligned": "MISALIGNED"
    }

    recommenders = {
        "optimized_estimate_maximize": lambda Q, n_agents: optimized_heuristic_recommender(Q, n_agents, method="estimate", minimize=False),
        "random": random_recommender,
        "none": constant_recommender,
        "aligned_heuristic": aligned_heuristic_recommender,
    }

    NAME = f"sweep_aligned_heuristic_e{epsilon}_q{sizeQinit}_N{N_AGENTS}_S{N_STATES}_A{N_ACTIONS}_I{N_ITER}_g{GAMMA}_a{ALPHA}_q{QINIT}"

    if not os.path.isdir(NAME):
        os.mkdir(NAME)

    results = []
    for i, e in enumerate(tqdm.tqdm(epsilons)):
        for norm, initTable in qinits.items():
            for recommender_type, recommender_function in recommenders.items():

                experiment_data = {}

                for t in range(0, N_REPEATS):
                    M = single_run(braess_augmented_network, N_AGENTS, N_STATES, N_ACTIONS, N_ITER, e, GAMMA,
                                   ALPHA, initTable, recommender_function)

                    experiment_data[t] = M

                    W = [M[t]["R"].mean() for t in range(0, N_ITER)]
                    L = nolds.lyap_r(W)
                    T = np.mean(W[int(0.8 * N_ITER):N_ITER])
                    T_all = np.mean(W)
                    T_std = np.std(W[int(0.8 * N_ITER):N_ITER])

                    groups = [M[t]["groups"] for t in range(0, N_ITER)]
                    groups_mean = np.mean(groups)
                    groups_var = np.var(groups)
                    Qvar = [M[t]["Qvar"] for t in range(0, N_ITER)]
                    Qvar_mean = np.mean(Qvar)

                    if recommender_type == "none":
                        alignment = None
                        alignment_all = None
                    else:
                        alignment = np.array([M[t]["alignment"][1] for t in range(int(0.8 * N_ITER), N_ITER)])
                        alignment_all = np.array([M[t]["alignment"][1] for t in range(N_ITER)]).mean(axis=0)
                        alignment = alignment.mean(axis=0)

                    row = {
                        "epsilon": e,
                        "norm": norm,
                        "T_mean": T,
                        "T_mean_all": T_all,
                        "T_std": T_std,
                        "Lyapunov": L,
                        "repetition": t,
                        "groups_mean": groups_mean,
                        "groups_var": groups_var,
                        "Qvar_mean": Qvar_mean,
                        "recommender_type": recommender_type,
                        "alignment": alignment,
                        "alignment_all": alignment_all
                    }

                    results.append(row)

                filename = f"{NAME}/run_e{e}_{norm}_{recommender_type}"
                with open(filename, "wb") as file:
                    pickle.dump(experiment_data, file)

    df = pd.DataFrame(results)

    df.to_csv(NAME + ".csv")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('epsilon', type=float)
    args = parser.parse_args()

    main(args.epsilon)
