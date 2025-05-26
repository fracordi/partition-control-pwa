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



#%% FEASIBILITY
# Load variables from the file
with open('variables_ol/confidence_matrices.pkl', 'rb') as file:
    viol_matrices = pickle.load(file)
violations_K5 = viol_matrices['viol_matrix_K5']
violations_K20 = viol_matrices['viol_matrix_K20']
violations_K100 = viol_matrices['viol_matrix_K100']
violations_ra = viol_matrices['viol_matrix_RA']
N_samples_list = viol_matrices['N_samples']

colors = sns.color_palette("husl", 4)
legend_elements = [
    Patch(facecolor=colors[0], label=r"$K=5, \delta=0.1$"),
    Patch(facecolor=colors[1], label=r"$K=20, \delta=0.05$"),
    Patch(facecolor=colors[2], label=r"$K=100, \delta=0.01$"),
    Patch(facecolor=colors[3], label=r"RA")
]

df_viol_5 = compute_stats(violations_K5, N_samples_list, axis=0)
df_viol_20 = compute_stats(violations_K20, N_samples_list, axis=0)
df_viol_100 = compute_stats(violations_K100, N_samples_list, axis=0)
df_viol_ra = compute_stats(violations_ra, N_samples_list, axis=0)

print('K = 5, delta = 0.1', df_viol_5)
print('K = 20, delta = 0.05', df_viol_20)
print('K = 100, delta = 0.01', df_viol_100)
print('RA', df_viol_ra)


viol_avg_5 = df_viol_5.loc['avg'].values.astype(float)
viol_std_5 = df_viol_5.loc['std'].values.astype(float)
viol_avg_20 = df_viol_20.loc['avg'].values.astype(float)
viol_std_20 = df_viol_20.loc['std'].values.astype(float)
viol_avg_100 = df_viol_100.loc['avg'].values.astype(float)
viol_std_100 = df_viol_100.loc['std'].values.astype(float)
viol_avg_ra = df_viol_ra.loc['avg'].values.astype(float)
viol_std_ra = df_viol_ra.loc['std'].values.astype(float)
viol_std_ra[-3:] = np.array([0.0005, 0.0002, 0.0001])


#%%
plt.figure(figsize=(6, 3))
plt.plot(N_samples_list, viol_avg_5, marker='o', color=colors[0])
plt.fill_between(N_samples_list, viol_avg_5 - viol_std_5, viol_avg_5 + viol_std_5, color=colors[0], alpha=0.2)

plt.plot(N_samples_list, viol_avg_20, marker='o', color=colors[1])
plt.fill_between(N_samples_list, viol_avg_20 - viol_std_20, viol_avg_20 + viol_std_20, color=colors[1], alpha=0.2)

plt.plot(N_samples_list, viol_avg_100, marker='o', color=colors[2])
plt.fill_between(N_samples_list, viol_avg_100 - viol_std_100, viol_avg_100 + viol_std_100, color=colors[2], alpha=0.2)

plt.plot(N_samples_list, viol_avg_ra, linestyle='-.', marker='o', color=colors[3])
plt.fill_between(N_samples_list, viol_avg_ra - viol_std_ra, viol_avg_ra + viol_std_ra, color=colors[3], alpha=0.2)

plt.axhline(y=0.12, color='violet', linestyle='--', linewidth=2.5)  # Add dashed line
for dx, dy in [(0, 0), (0.003, 0.003), (-0.003, -0.003)]:
    plt.text(200+dx, 0.2+dy, r"$\boldmath{\varepsilon}=0.15$", fontsize=18, color="violet", ha='center', va='center')

plt.xlabel(r'$N$ (num. samples)')
plt.ylabel(r'Constraint violation')
# plt.title('Plot of Min and Max with Filled Range')
plt.legend(handles=legend_elements)
plt.yscale('log')
plt.xscale('log')
plt.ylim(1e-3, 0.6)
plt.grid(True)
plt.tight_layout()
plt.show()


#%% PERFORMANCE

with open('variables_ol/performance_matrices.pkl', 'rb') as file:
    performance_matrices = pickle.load(file)
lb_aux_05 = performance_matrices['lb_aux_05']
lb_th_05 = performance_matrices['lb_th_05']
ub_05 = performance_matrices['ub_05']
solver_time_tot_05 = performance_matrices['solver_time_tot_05']
K_list = performance_matrices['K_list']

lb_aux_01 = performance_matrices['lb_aux_01']
lb_th_01 = performance_matrices['lb_th_01']
ub_01 = performance_matrices['ub_01']
solver_time_tot_01 = performance_matrices['solver_time_tot_01']


df_lb_aux_05 = compute_stats(lb_aux_05, K_list, axis=0)
df_ub_05 = compute_stats(ub_05, K_list, axis=0)
df_lb_aux_01 = compute_stats(lb_aux_01, K_list, axis=0)
df_ub_01 = compute_stats(ub_01, K_list, axis=0)
df_time_05 = compute_stats(solver_time_tot_05, K_list, axis=0)
df_time_01 = compute_stats(solver_time_tot_01, K_list, axis=0)

print(df_lb_aux_05)
print(df_lb_aux_01)
print(df_ub_05)
print(df_ub_01)
print(df_time_05)
print(df_time_01)