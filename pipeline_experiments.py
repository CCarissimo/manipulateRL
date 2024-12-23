import numpy as np
from tqdm.auto import tqdm
import pickle
import nolds
import pandas as pd
from recommenders import random_recommender, constant_recommender, optimized_heuristic_recommender, aligned_heuristic_recommender
from single_run import single_run
from routing_networks import braess_augmented_network
import os
import multiprocessing as mp


def get_unique_filename(base_filename):
    if not os.path.exists(base_filename):
        return base_filename

    filename, ext = os.path.splitext(base_filename)
    index = 1
    while True:
        new_filename = f"{filename}_{index}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        index += 1


def save_pickle_with_unique_filename(data, filename):
    unique_filename = get_unique_filename(filename)
    with open(unique_filename, 'wb') as file:
        pickle.dump(data, file)


def parallel_function(path, n_agents, n_states, recommender_type, recommender_function):
    # Base Settings Which Will Not Change
    N_ACTIONS = 3
    N_ITER = 10000
    N_REPEATS = 40
    EPSILON = "DECAYED"
    GAMMA = 0
    ALPHA = 0.1
    initTable = "UNIFORM"

    path_to_experiment = f"{path}/n{n_agents}_s{n_states}_rec{recommender_type}"

    if not os.path.isdir(experiment):
        os.mkdir(experiment)

    results = []
    for t in range(0, N_REPEATS):
        M = single_run(braess_augmented_network, n_agents, n_states, N_ACTIONS, N_ITER, EPSILON, GAMMA,
                       ALPHA, initTable, recommender_function)

        W = [M[t]["R"].mean() for t in range(0, N_ITER)]

        filename = get_unique_filename(f"{path_to_experiment}/timeseries.npy")
        np.save(filename, np.array(W))

        L = nolds.lyap_r(W)
        T = np.mean(W[int(0.8 * N_ITER):N_ITER])
        T_all = np.mean(W)
        T_std = np.std(W[int(0.8 * N_ITER):N_ITER])

        # groups = [M[t]["groups"] for t in range(0, N_ITER)]
        # groups_mean = np.mean(groups)
        # groups_var = np.var(groups)
        Qvar = [M[t]["Qvar"] for t in range(0, N_ITER)]
        Qvar_mean = np.mean(Qvar)

        # if recommender_type == "none":
        #     alignment = None
        #     alignment_all = None
        # else:
        #     alignment = np.array([M[t]["alignment"][1] for t in range(int(0.8 * N_ITER), N_ITER)])
        #     alignment_all = np.array([M[t]["alignment"][1] for t in range(N_ITER)]).mean(axis=0)
        #     alignment = alignment.mean(axis=0)

        row = {
            "T_mean": T,
            "T_mean_all": T_all,
            "T_std": T_std,
            "Lyapunov": L,
            "repetition": t,
            # "groups_mean": groups_mean,
            # "groups_var": groups_var,
            "Qvar_mean": Qvar_mean,
            "recommender_type": recommender_type,
            # "alignment": alignment,
            # "alignment_all": alignment_all
        }

        results.append(row)

    return results


def run_apply_async_multiprocessing(func, argument_list, num_processes):
    pool = mp.Pool(processes=num_processes)

    jobs = [
        pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func,
                                                                                                            args=(
                                                                                                            argument,))
        for argument in argument_list]
    pool.close()
    result_list_tqdm = []
    for job in tqdm(jobs):
        result_list_tqdm.append(job.get())

    return result_list_tqdm


def heuristic_estimate_maximize(Q, n_agents):
    return optimized_heuristic_recommender(Q, n_agents, method="estimate", minimize=False)


def main(path):  # pass epsilon as an argument using argparse

    # Base Settings Which Will Not Change
    N_AGENTS = 100
    N_STATES = 3
    N_ACTIONS = 3
    N_ITER = 10000
    mask = np.zeros(N_AGENTS)
    mask[:] = 1
    GAMMA = 0
    ALPHA = 0.1
    epsilon = "DECAYED"

    # Parameters which will be Varied
    numbers_of_states = np.arange(3, 100, 10)
    numbers_of_agents = np.arange(100, 1000, 100)
    recommenders = {
        "optimized_estimate_maximize": heuristic_estimate_maximize,
        "random": random_recommender,
        "none": constant_recommender,
        # "aligned_heuristic": aligned_heuristic_recommender,
    }

    NAME = f"{path}/sweep_size_e{epsilon}_qUNIFORM_Nvariable_Svariable_A{N_ACTIONS}_I{N_ITER}_g{GAMMA}_a{ALPHA}"

    if not os.path.isdir(NAME):
        os.mkdir(NAME)

    num_cpus = int(os.environ.get("SLURM_NTASKS", os.cpu_count()))  # specific for euler cluster
    argument_list = []
    for n_agents in numbers_of_agents:
        for n_states in numbers_of_states:
            for recommender_type, recommender_function in recommenders.items():
                parameter_tuple = (path, n_agents, n_states, recommender_type, recommender_function)
                argument_list.append(parameter_tuple)
    results = run_apply_async_multiprocessing(parallel_function, argument_list=argument_list, num_processes=num_cpus)

    save_pickle_with_unique_filename(results, f"{NAME}/results.pkl")
    df = pd.DataFrame(results)
    df.to_csv(NAME+"/results.csv")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    main(args.path)
