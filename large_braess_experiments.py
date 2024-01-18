import numpy as np
from tqdm.auto import tqdm
import pickle
import pandas as pd
from recommenders import random_recommender, constant_recommender, optimized_heuristic_recommender, aligned_heuristic_recommender
from single_run_large_braess import single_run
import os
import multiprocessing as mp
from dataclasses import dataclass


@dataclass
class RecommenderExperimentConfig:
    path: str
    n_agents: int
    n_states: int
    recommender: str


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


def parallel_function(config):
    # Base Settings Which Will Not Change
    N_ACTIONS = 20
    N_ITER = 10000
    N_REPEATS = 40
    EPSILON = "DECAYED"
    GAMMA = 0
    ALPHA = 0.1
    initTable = "UNIFORM"

    path_to_experiment = f"{config.path}/n{config.n_agents}_s{config.n_states}_{config.recommender}"

    if not os.path.isdir(path_to_experiment):
        os.mkdir(path_to_experiment)

    results = []
    for t in range(0, N_REPEATS):
        M = single_run(config.n_agents, config.n_states, N_ACTIONS, N_ITER, EPSILON, GAMMA,
                       ALPHA, initTable, config.recommender)

        W = [M[t]["R"].mean() for t in range(0, N_ITER)]

        filename = get_unique_filename(f"{path_to_experiment}/timeseries.npy")
        np.save(filename, np.array(W))

        # L = nolds.lyap_r(W)
        T = np.mean(W[int(0.8 * N_ITER):N_ITER])
        T_all = np.mean(W)
        T_std = np.std(W[int(0.8 * N_ITER):N_ITER])

        # groups = [M[t]["groups"] for t in range(0, N_ITER)]
        # groups_mean = np.mean(groups)
        # groups_var = np.var(groups)
        Qvar = [M[t]["Qvar"] for t in range(0, N_ITER)]
        Qvar_mean = np.mean(Qvar)

        # if config.recommender_type == "none":
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
            # "Lyapunov": L,
            "repetition": t,
            # "groups_mean": groups_mean,
            # "groups_var": groups_var,
            "Qvar_mean": Qvar_mean,
            "recommender_type": config.recommender,
            "n_states": config.n_states,
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


def main(path):  # pass epsilon as an argument using argparse

    # Base Settings Which Will Not Change
    N_AGENTS = 100
    N_STATES = 20
    N_ACTIONS = 20
    N_ITER = 10000
    GAMMA = 0
    ALPHA = 0.1
    epsilon = "DECAYED"

    # Parameters which will be Varied
    numbers_of_states = np.arange(20, 200, 10)
    # numbers_of_agents = np.arange(100, 1000, 100)
    recommenders = {
        "optimized": 0,
        "random": 1,
        "constant": 2,
    }

    NAME = f"{path}/large_braess_recommendation_sweep"

    if not os.path.isdir(NAME):
        os.mkdir(NAME)

    num_cpus = int(os.environ.get("SLURM_NTASKS", os.cpu_count()))  # specific for euler cluster
    argument_list = []
    for n_states in numbers_of_states:
        for recommender in recommenders.keys():
            if n_states > 20 and recommender == "constant":
                continue
            config = RecommenderExperimentConfig(path, N_AGENTS, n_states, recommender)
            argument_list.append(config)
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
