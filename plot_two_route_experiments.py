import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval

######### DATA IMPORTS AND PREPROCESSING
path = "experiments/two_route_experiments_2.csv"

df = pd.read_csv(path)

print(df["alignment"])
df["alignment"] = df["alignment"].astype("float64")

means = df.groupby(["recommender_type", "epsilon"]).mean()
std = df.groupby(["recommender_type", "epsilon"]).std()


######### END OF DATA IMPORTS AND PREPROCESSING


def welfare(epsilon):
    return - (2 - 2 / 3 * epsilon + 2 / 9 * epsilon ** 2)


def plot_case(axis, recommender, quantity, linestyle):
    means.loc[recommender][quantity].plot(label=recommender, ax=axis, linestyle=linestyle, legend=False)
    axis.fill_between(np.linspace(0, 1, 51),
                      means.loc[recommender][quantity] + std.loc[recommender][quantity],
                      means.loc[recommender][quantity] - std.loc[recommender][quantity],
                      alpha=0.2)


opt = np.ones(21) * -1.5
x_vals = np.linspace(0, 1, 51)
y_vals = welfare(x_vals)

sns.set_context("paper")
# plt.style.use('paper.mplstyle')
# cmap = plt.get_cmap('plasma')
sns.set_palette("magma")
# colors = [cmap(c) for c in np.linspace(0.1, 0.9, n_actions)]

initialization = "uniform"
metric_mean = "T_mean_all"
metric_alignment = "alignment_all"

fig, [ax, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(6.75, 3.2))
sns.despine()

plot_case(ax, "constant_recommender", metric_mean, linestyle="-")
plot_case(ax, "random_recommender", metric_mean, linestyle="-")
plot_case(ax, "misaligned_recommender", metric_mean, linestyle=(0, (1, 5)))
plot_case(ax, "aligned_recommender", metric_mean, linestyle=(0, (1, 3)))

ax.set_ylim((-1.55, -1.49))
# ax.set_yticks(ticks=np.linspace(-1.55, -1.6, 5), labels=np.linspace(-1.55, -1.6, 5))

ax.set_xticks(ticks=np.linspace(0, 1, 5))

# ax.plot(x_vals, opt, linestyle="--", color="gray", linewidth=)
ax.axhline(-1.5, linestyle="--", color="gray", linewidth=0.5)
ax.annotate('social optimum', xy=(0.65, 0.84), xycoords='axes fraction')
# ax.plot(x_vals, opt-0.5, linestyle="--", color="gray")
# ax.axhline(-2, linestyle="--", color="gray", linewidth=0.5)
# ax.annotate('nash equilibrium', xy=(0.505, 0.925), xycoords='axes fraction')
# ax.plot(x_vals, y_vals, label="irrational agents", linestyle=":", color="black")
ax.set_xlabel(r"exploration rate ($\epsilon$)", **{"fontname": "Times New Roman", "fontsize": "x-large"})
ax.set_ylabel(r"social welfare", **{"fontname": "Times New Roman", "fontsize": "x-large"})
# ax.legend(prop={"family": "Times New Roman", "size": "x-large"}, title="recommender type", bbox_to_anchor=(1.05, 1))
# plt.savefig("plots/recommenders_welfare_comparison.pdf")
# plt.show()

plot_case(ax2, "constant_recommender", metric_alignment, linestyle="-")
# plot_case(ax2, "naive", metric_alignment, linestyle="-")
plot_case(ax2, "random_recommender", metric_alignment, linestyle="--")
# plot_case(ax2, "heuristic", metric_alignment, linestyle="-")
# plot_case(ax2, "optimized_action_maximize", metric_alignment, linestyle="--")
plot_case(ax2, "misaligned_recommender", metric_alignment, linestyle=(0, (1, 5)))
plot_case(ax2, "aligned_recommender", metric_alignment, linestyle=(0, (1, 3)))

ax2.set_ylim((-0.05, 1.05))
ax2.set_xticks(ticks=np.linspace(0, 1, 5))

ax2.set_xlabel(r"exploration rate ($\epsilon$)", **{"fontname": "Times New Roman", "fontsize": "x-large"})
ax2.set_ylabel(r"recommendation alignment", **{"fontname": "Times New Roman", "fontsize": "x-large"})
# ax2.legend(prop={"family": "Times New Roman", "size": "x-large"}, title="recommender type", bbox_to_anchor=(1.05, 1))

handles, labels = ax.get_legend_handles_labels()
print(handles)
print(labels)
labels[0] = "constant"
labels[1] = "random"
labels[2] = "misaligned"
labels[3] = "aligned"

fig.legend(bbox_to_anchor=(0.5, 0.95), loc='lower center', ncol=4, handles=handles, labels=labels)
fig.tight_layout()
fig.savefig(f"plots/recommenders_{metric_mean}_{metric_alignment}.pdf", bbox_inches="tight")
plt.show()
