import glob
import numpy as np
import pandas as pd

def main(path):
	results = []
	for folder in glob.glob(path + "/*"):

		# f"{path}/n{n_agents}_s{n_states}_rec{recommender_type}"
		split = folder.split("_")
		n_states = split[1].split("s")[-1]
		n_agents = split[0].split("n")[-1]
		recommender_type = split[2].split("rec")[-1]

		for file in glob.glob(path + "/" + folder + "/*"):
			filepath = path + "/" + folder + "/*" + file
			timeseries = np.load(filepath)

			T = np.mean(timeseries[int(0.8 * len(timeseries)):len(timeseries)])
        	T_all = np.mean(timeseries)
        	T_std = np.std(timeseries[int(0.8 * len(timeseries)):len(timeseries)])

        	row = {
	        	"recommender_type": recommender_type,
	        	"n_agents": n_agentsn,
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
