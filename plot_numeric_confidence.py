from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from matplotlib.lines import Line2D
from matplotlib import rc
from matplotlib.patches import Patch
from utils import compute_stats


sns.set_theme()
TINY_SIZE = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 18
rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the legend
rc('legend', fontsize=TINY_SIZE)  # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc["font"] = "serif"
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

# plt.rc('font', family='serif')
mrk_size = 9


#%% Feasibility data

# Load variables from the file
with open('variables_numerical/numeric_confidence_partition_nx10_nunc10_K25_delta0.1_N_exp_confidence500.pkl', 'rb') as file:
    variables_K25 = pickle.load(file)
violations_K25 = variables_K25['empirical_violation']

with open('variables_numerical/numeric_confidence_partition_nx10_nunc10_K100_delta0.01_N_exp_confidence500.pkl', 'rb') as file:
    variables_K100 = pickle.load(file)
violations_K100 = variables_K100['empirical_violation']

with open('variables_numerical/numeric_confidence_random_nx10_nunc10_N_exp_confidence500.pkl', 'rb') as file:
    variables_ra_1 = pickle.load(file)
violations_ra_1 = variables_ra_1['empirical_violation']

with open('variables_numerical/numeric_confidence_random_nx10_nunc10_N_exp_confidence500_only_5000.pkl', 'rb') as file:
    variables_ra_2 = pickle.load(file)
violations_ra_2 = variables_ra_2['empirical_violation']

with open('variables_numerical/numeric_confidence_random_nx10_nunc10_N_exp_confidence500_only_10000.pkl', 'rb') as file:
    variables_ra_3 = pickle.load(file)
violations_ra_3 = variables_ra_3['empirical_violation']

# Update data with N=5000, 10000 for RA, since they have been simulated separately
violations_ra_1[9, :] = violations_ra_2
violations_ra_1[10, :] = violations_ra_3

violations_ra = violations_ra_1

colors = sns.color_palette("husl", 3)

legend_elements = [
    Patch(facecolor=colors[0], label=r"$K=25, \delta=0.1$"),
    Patch(facecolor=colors[1], label=r"$K=100, \delta=0.01$"),
    Patch(facecolor=colors[2], label=r"RA"),
    Line2D([0], [0],
           color='violet',
           linestyle='--',
           linewidth=1.5,
           label=r"$\varepsilon$")
]

N_samples_list = variables_K25['N_samples_list']
df_viol_25 = compute_stats(violations_K25, N_samples_list)
df_viol_100 = compute_stats(violations_K100, N_samples_list)
df_viol_ra = compute_stats(violations_ra, N_samples_list)

print('K = 25, delta = 0.1', df_viol_25)
print('K = 100, delta = 0.01', df_viol_100)
print('RA', df_viol_ra)


viol_avg_25 = df_viol_25.loc['avg'].values.astype(float)
viol_std_25 = df_viol_25.loc['std'].values.astype(float)
viol_avg_100 = df_viol_100.loc['avg'].values.astype(float)
viol_std_100 = df_viol_100.loc['std'].values.astype(float)
viol_avg_ra = df_viol_ra.loc['avg'].values.astype(float)
viol_std_ra = df_viol_ra.loc['std'].values.astype(float)

viol_max_25 = df_viol_25.loc['max'].values.astype(float)
viol_max_100 = df_viol_100.loc['max'].values.astype(float)
viol_max_ra = df_viol_ra.loc['max'].values.astype(float)


#%%
plt.figure(figsize=(6, 3))
plt.plot(N_samples_list, viol_avg_25, marker='o', color=colors[0])
plt.fill_between(N_samples_list, viol_avg_25 - viol_std_25, viol_avg_25 + viol_std_25, color=colors[0], alpha=0.2)
plt.plot(N_samples_list, viol_max_25, marker='*', markersize=7, color=colors[0], linestyle='None')


plt.plot(N_samples_list, viol_avg_100, marker='o', color=colors[1])
plt.fill_between(N_samples_list, viol_avg_100 - viol_std_100, viol_avg_100 + viol_std_100, color=colors[1], alpha=0.2)
plt.plot(N_samples_list, viol_max_100, marker='*', markersize=7, color=colors[1], linestyle='None')


plt.plot(N_samples_list, viol_avg_ra, linestyle='-.', marker='o', color=colors[2])
plt.fill_between(N_samples_list, viol_avg_ra - viol_std_ra, viol_avg_ra + viol_std_ra, color=colors[2], alpha=0.2)
plt.plot(N_samples_list, viol_max_ra, marker='*', markersize=7, color=colors[2], linestyle='None')

plt.axhline(y=0.15, color='violet', linestyle='--', linewidth=2.5)  # Add dashed line
# for dx, dy in [(0, 0), (0.0005, 0.0005), (-0.0005, -0.0005), (-0.0005, 0.0005), (0.0005, -0.0005)]:
#     plt.text(200+dx, 0.17+dy, r"$\boldmath{\varepsilon}=0.15$", fontsize=18, color="violet", ha='center', va='center')

plt.xlabel(r'$N$ (num. samples)')
plt.ylabel(r'Emp. constraint violation')
# plt.yscale('log')
plt.xscale('log')
plt.ylim(-0.002, 0.2)

leg = plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)
leg.texts[-1].set_fontsize(11)
plt.grid(True)
plt.tight_layout()
plt.show()

