import glob
import pickle

import numpy as np
import pandas as pd


def keep_timeseries(path):
    results = dict()
    for folder in glob.glob(path + "/n*"):

        # f"{path}/n{n_agents}_s{n_states}_rec{recommender_type}"
        split = folder.split("/")[-1].split("_")
        n_states = split[1].replace("s", "")
        n_agents = split[0].replace("n", "")
        recommender_type = split[2].replace("rec", "")

        print(recommender_type, n_states, n_agents)

        T = []
        for file in glob.glob(folder + "/*"):
            timeseries = np.load(file)
            T.append(timeseries)
        timeseries = np.vstack(T)
        timeseries_mean = timeseries.mean(axis=0)
        timeseries_median = np.percentile(timeseries, 50, axis=0)
        timeseries_p25 = np.percentile(timeseries, 25, axis=0)
        timeseries_p75 = np.percentile(timeseries, 75, axis=0)

        d = {
            "mean": timeseries_mean,
            "median": timeseries_median,
            "percentile25": timeseries_p25,
            "percentile75": timeseries_p75
        }

        results[(recommender_type, n_agents, n_states)] = d

    with open(path + "/timeseries.pkl", "wb") as file:
        pickle.dump(results, file)


def main(path):
    results = []
    for folder in glob.glob(path + "/n*"):

        # f"{path}/n{n_agents}_s{n_states}_rec{recommender_type}"
        split = folder.split("/")[-1].split("_")
        n_states = split[1].replace("s", "")
        n_agents = split[0].replace("n", "")
        recommender_type = split[2].replace("rec", "")

        print(recommender_type, n_states, n_agents)

        for file in glob.glob(folder + "/*"):
            timeseries = np.load(file)

            T = np.mean(timeseries[int(0.8 * len(timeseries)):len(timeseries)])
            T_all = np.mean(timeseries)
            T_std = np.std(timeseries[int(0.8 * len(timeseries)):len(timeseries)])

            row = {
                "recommender_type": recommender_type,
                "n_agents": n_agents,
                "n_states": n_states,
                "T_mean": T,
                "T_mean_all": T_all,
                "T_std": T_std,
            }
            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(path + "/dataframe.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    main(args.path)
