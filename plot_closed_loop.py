import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from matplotlib import rc
from matplotlib.patches import Patch
from utils import compute_stats


sns.set_theme()
TINY_SIZE = 9
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 18
rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('legend', fontsize=TINY_SIZE)  # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc["font"] = "serif"
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

# plt.rc('font', family='serif')
mrk_size = 9



#%%
# Load variables from the file
with open('variables_cl/cost_matrices.pkl', 'rb') as file:
    cost_matrices = pickle.load(file)
cost_matrix_adpt_4 = cost_matrices['cost_matrix_adpt_4']
cost_matrix_adpt_8 = cost_matrices['cost_matrix_adpt_8']
cost_matrix_kmns_4 = cost_matrices['cost_matrix_kmns_4']
cost_matrix_kmns_8 = cost_matrices['cost_matrix_kmns_8']
T_cl = cost_matrix_adpt_4.shape[1]

stats_adpt_4 = compute_stats(cost_matrix_adpt_4, np.arange(T_cl), axis=0)
stats_adpt_8 = compute_stats(cost_matrix_adpt_8, np.arange(T_cl), axis=0)
stats_kmns_4 = compute_stats(cost_matrix_kmns_4, np.arange(T_cl), axis=0)
stats_kmns_8 = compute_stats(cost_matrix_kmns_8, np.arange(T_cl), axis=0)

mean_adpt_4 = stats_adpt_4.loc['avg'].values.astype(float)
mean_adpt_8 = stats_adpt_8.loc['avg'].values.astype(float)
mean_kmns_4 = stats_kmns_4.loc['avg'].values.astype(float)
mean_kmns_8 = stats_kmns_8.loc['avg'].values.astype(float)

std_adpt_4 = stats_adpt_4.loc['std'].values.astype(float)
std_adpt_8 = stats_adpt_8.loc['std'].values.astype(float)
std_kmns_4 = stats_kmns_4.loc['std'].values.astype(float)
std_kmns_8 = stats_kmns_8.loc['std'].values.astype(float)

min_adpt_4 = stats_adpt_4.loc['min'].values.astype(float)
min_adpt_8 = stats_adpt_8.loc['min'].values.astype(float)
min_kmns_4 = stats_kmns_4.loc['min'].values.astype(float)
min_kmns_8 = stats_kmns_8.loc['min'].values.astype(float)

max_adpt_4 = stats_adpt_4.loc['max'].values.astype(float)
max_adpt_8 = stats_adpt_8.loc['max'].values.astype(float)
max_kmns_4 = stats_kmns_4.loc['max'].values.astype(float)
max_kmns_8 = stats_kmns_8.loc['max'].values.astype(float)

time_range = np.arange(T_cl)

colors = sns.color_palette("husl", 4)

legend_elements = [
    Patch(facecolor=colors[0], label=r"ADPT, $K=4$"),
    Patch(facecolor=colors[1], label=r"ADPT, $K=8$"),
    Patch(facecolor=colors[2], label=r"KMNS, $K=4$"),
    Patch(facecolor=colors[3], label=r"KMNS, $K=8$")
]

# Cost per time
scale_std = 1
plt.figure(figsize=(6, 3))
plt.plot(time_range, mean_adpt_4, color=colors[0])
plt.fill_between(time_range, np.maximum(mean_adpt_4 - scale_std * std_adpt_4, 0.1), mean_adpt_4 + scale_std * std_adpt_4, color=colors[0], alpha=0.2)
# plt.fill_between(time_range, min_adpt_4, max_adpt_4, color=colors[0], alpha=0.2)

plt.plot(time_range, mean_adpt_8, color=colors[1])
plt.fill_between(time_range, np.maximum(mean_adpt_8 - scale_std * std_adpt_8, 0.1), mean_adpt_8 + scale_std * std_adpt_8, color=colors[1], alpha=0.2)
# plt.fill_between(time_range, min_adpt_8, max_adpt_8, color=colors[1], alpha=0.2)

plt.plot(time_range, mean_kmns_4, color=colors[2])
plt.fill_between(time_range, np.maximum(mean_kmns_4 - scale_std * std_kmns_4, 0.1), mean_kmns_4 + scale_std * std_kmns_4, color=colors[2], alpha=0.2)
# plt.fill_between(time_range, min_kmns_4, max_kmns_4, color=colors[2], alpha=0.2)

plt.plot(time_range, mean_kmns_8, color=colors[3])
plt.fill_between(time_range, np.maximum(mean_kmns_8 - scale_std * std_kmns_8, 0.1), mean_kmns_8 + scale_std * std_kmns_8, color=colors[3], alpha=0.2)
# plt.fill_between(time_range, min_kmns_8, max_kmns_8, color=colors[3], alpha=0.2)


plt.xlabel(r'Time step')
plt.ylabel(r'$\ell_{\mathrm{cl}}(t)$')
# plt.title('Plot of Min and Max with Filled Range')
plt.legend(handles=legend_elements, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False)
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.show()
