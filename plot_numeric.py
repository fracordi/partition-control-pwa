import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from matplotlib import rc
from matplotlib.patches import Patch

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

#%% LOW DIM UNC

variables_string_list = ['_nx10', '_nx20', '_nx30', '_nx40', '_nx50']

# PP n_unc = 5
violation_pp_nunc5 = []
for nx in variables_string_list:
    file_to_open = 'variables_numerical/numeric_partition' + nx + '_nunc5_K50.pkl'
    with open(file_to_open, 'rb') as file:
        dict_results = pickle.load(file)
    violations = dict_results['empirical_violation']
    violation_feas = violations[violations>=0]
    violation_avg = np.mean(violation_feas)
    violation_pp_nunc5.append(violation_avg)

# PP n_unc = 5
violation_pp_nunc10 = []
for nx in variables_string_list:
    file_to_open = 'variables_numerical/numeric_partition' + nx + '_nunc10_K50.pkl'
    with open(file_to_open, 'rb') as file:
        dict_results = pickle.load(file)
    violations = dict_results['empirical_violation']
    violation_feas = violations[violations>=0]
    violation_avg = np.mean(violation_feas)
    violation_pp_nunc10.append(violation_avg)

# RA n_unc = 5
violation_ra_nunc5 = []
for nx in variables_string_list:
    file_to_open = 'variables_numerical/numeric_random' + nx + '_nunc5.pkl'
    with open(file_to_open, 'rb') as file:
        dict_results = pickle.load(file)
    violations = dict_results['empirical_violation']
    violation_feas = violations[violations>=0]
    violation_avg = np.mean(violation_feas)
    violation_ra_nunc5.append(violation_avg)

# RA n_unc = 5
violation_ra_nunc10 = []
for nx in variables_string_list:
    file_to_open = 'variables_numerical/numeric_random' + nx + '_nunc10.pkl'
    with open(file_to_open, 'rb') as file:
        dict_results = pickle.load(file)
    violations = dict_results['empirical_violation']
    violation_feas = violations[violations>=0]
    violation_avg = np.mean(violation_feas)
    violation_ra_nunc10.append(violation_avg)

# Green colors
green_light = (0.56, 0.85, 0.60)
green_dark  = (0.13, 0.65, 0.32)

# Purple colors
purple_light = (0.78, 0.60, 0.95)
purple_dark  = (0.45, 0.25, 0.75)

colors = [green_light, purple_light , green_dark , purple_dark]

legend_elements = [
    Patch(facecolor=colors[0], label=r"PP, $n_\theta=5$"),
    Patch(facecolor=colors[1], label=r"RA, $n_\theta=5$"),
    Patch(facecolor=colors[2], label=r"PP, $n_\theta=10$"),
    Patch(facecolor=colors[3], label=r"RA, $n_\theta=10$")
]

n_x_list = [10, 20, 30, 40, 50]

plt.figure(figsize=(6, 2.7))
plt.plot(n_x_list, violation_pp_nunc5, marker='o', color=colors[0])
plt.plot(n_x_list, violation_ra_nunc5, marker='o', color=colors[1])
plt.plot(n_x_list, violation_pp_nunc10, marker='o', color=colors[2])
plt.plot(n_x_list, violation_ra_nunc10, marker='o', color=colors[3])

plt.xlabel(r'$n_x$')
plt.ylabel(r'Emp. constraint violation')
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)
# plt.yscale('log')
# plt.xscale('log')
# plt.ylim(1e-3, 0.6)
plt.xticks([10, 20, 30, 40, 50])
plt.grid(True)
plt.tight_layout()
plt.show()

