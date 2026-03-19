import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from matplotlib import rc


def double_axis_boxplot(group_arrays, group_labels, axis1_ylim=None, axis2_ylim=None,
                        axis1_label='label1', axis2_label='label2', n_ticks=4,
                        dx=0.18, box_width=0.22, showfliers=True, figsize=(7, 4), dpi=150):
    """
    Function for boxplots for two axis
    """
    colors = sns.husl_palette( n_colors=2, s=1.5)


    n_groups = len(group_arrays)

    # Extract data for the two axis
    axis1_data, axis2_data = [], []
    for arr in group_arrays:
        arr = np.asarray(arr)
        axis1_data.append(arr[0].ravel())
        axis2_data.append(arr[1].ravel())

    x = np.arange(1, n_groups + 1)
    pos_axis1 = x - dx
    pos_axis2 = x + dx

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    ax2 = ax1.twinx()

    # Axis 1 boxplots (left axis)
    bp1 = ax1.boxplot(axis1_data, positions=pos_axis1, widths=box_width, patch_artist=True,
                      showfliers=showfliers, manage_ticks=False)

    # Axis 2 boxplots (right axis)
    bp2 = ax2.boxplot(axis2_data, positions=pos_axis2, widths=box_width, patch_artist=True,
                      showfliers=showfliers, manage_ticks=False)

    # Plot colors and style
    bp_list = [bp1, bp2]
    for i, bp in enumerate(bp_list):
        facecolor = edgecolor = colors[i]
        for box in bp["boxes"]:
            box.set_facecolor(facecolor)
            box.set_alpha(0.25)
            box.set_edgecolor(edgecolor)
            box.set_linewidth(1.2)
        for k in ["whiskers", "caps", "medians"]:
            for line in bp[k]:
                line.set_color(edgecolor)
                line.set_linewidth(1.2)

    # Axis labels and ticks colors
    ax1.set_ylabel(axis1_label, color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax2.set_ylabel(axis2_label, color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    # X ticks defined by labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_labels)

    # Y limits
    if axis1_ylim is not None:
        ax1.set_ylim(axis1_ylim)
    if axis2_ylim is not None:
        ax2.set_ylim(axis2_ylim)

    # Set y ticks
    ax1.set_yticks(np.linspace(*ax1.get_ylim(), n_ticks))
    ax2.set_yticks(np.linspace(*ax2.get_ylim(), n_ticks))
    ax1.grid(axis="y", alpha=0.4)
    ax2.grid(axis="y", alpha=0.4)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



sns.set_theme()
TINY_SIZE = 11
SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
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
with open('variables_cl/closed_loop_adpt_gauss_KtoSplit3_delta0.05_Nexp500_Tcl30_1.pkl', 'rb') as file:
    adpt_3_all_vars = pickle.load(file)

with open('variables_cl/closed_loop_adpt_gauss_KtoSplit6_delta0.05_Nexp500_Tcl30_1.pkl', 'rb') as file:
    adpt_6_all_vars = pickle.load(file)

with open('variables_cl/closed_loop_adpt_gauss_KtoSplit9_delta0.05_Nexp500_Tcl30_1.pkl', 'rb') as file:
    adpt_9_all_vars = pickle.load(file)

with open('variables_cl/closed_loop_kmns_gauss_KtoSplit3_delta0.05_Nexp500_Tcl30_1.pkl', 'rb') as file:
    kmns_3_all_vars = pickle.load(file)

with open('variables_cl/closed_loop_kmns_gauss_KtoSplit6_delta0.05_Nexp500_Tcl30_1.pkl', 'rb') as file:
    kmns_6_all_vars = pickle.load(file)

with open('variables_cl/closed_loop_kmns_gauss_KtoSplit9_delta0.05_Nexp500_Tcl30_1.pkl', 'rb') as file:
    kmns_9_all_vars = pickle.load(file)

# Extract cost and time
cost_adpt_3 = adpt_3_all_vars['cost_matrix']
cost_adpt_6 = adpt_6_all_vars['cost_matrix']
cost_adpt_9 = adpt_9_all_vars['cost_matrix']
cost_kmns_3 = kmns_3_all_vars['cost_matrix']
cost_kmns_6 = kmns_6_all_vars['cost_matrix']
cost_kmns_9 = kmns_9_all_vars['cost_matrix']

time_adpt_3 = adpt_3_all_vars['solver_time_matrix']
time_adpt_6 = adpt_6_all_vars['solver_time_matrix']
time_adpt_9 = adpt_9_all_vars['solver_time_matrix']
time_kmns_3 = kmns_3_all_vars['solver_time_matrix']
time_kmns_6 = kmns_6_all_vars['solver_time_matrix']
time_kmns_9 = kmns_9_all_vars['solver_time_matrix']

print('AVG over time and exp. KMNS:', cost_kmns_3.mean(), cost_kmns_6.mean(), cost_kmns_9.mean())

# Select data during transient (for time: select till 10 as K=3,6,9 for sure)
cost_avgT_adpt_3 = cost_adpt_3[:, 0:12].mean(axis=1)
cost_avgT_adpt_6 = cost_adpt_6[:, 0:12].mean(axis=1)
cost_avgT_adpt_9 = cost_adpt_9[:, 0:12].mean(axis=1)
cost_avgT_kmns_3 = cost_kmns_3[:, 0:12].mean(axis=1)
cost_avgT_kmns_6 = cost_kmns_6[:, 0:12].mean(axis=1)
cost_avgT_kmns_9 = cost_kmns_9[:, 0:12].mean(axis=1)

time_avgT_adpt_3 = time_adpt_3[:, 0:10].mean(axis=1)
time_avgT_adpt_6 = time_adpt_6[:, 0:10].mean(axis=1)
time_avgT_adpt_9 = time_adpt_9[:, 0:10].mean(axis=1)
time_avgT_kmns_3 = time_kmns_3[:, 0:10].mean(axis=1)
time_avgT_kmns_6 = time_kmns_6[:, 0:10].mean(axis=1)
time_avgT_kmns_9 = time_kmns_9[:, 0:10].mean(axis=1)

# Stack together
data_time_transient_adpt_3 = np.vstack((cost_avgT_adpt_3, time_avgT_adpt_3))
data_time_transient_adpt_6 = np.vstack((cost_avgT_adpt_6, time_avgT_adpt_6))
data_time_transient_adpt_9 = np.vstack((cost_avgT_adpt_9, time_avgT_adpt_9))
data_time_transient_kmns_3 = np.vstack((cost_avgT_kmns_3, time_avgT_kmns_3))
data_time_transient_kmns_6 = np.vstack((cost_avgT_kmns_6, time_avgT_kmns_6))
data_time_transient_kmns_9 = np.vstack((cost_avgT_kmns_9, time_avgT_kmns_9))

print(time_avgT_adpt_3.mean(),
      time_avgT_adpt_6.mean(),
      time_avgT_adpt_9.mean(),
      time_avgT_kmns_3.mean(),
      time_avgT_kmns_6.mean(),
      time_avgT_kmns_9.mean())


data = [data_time_transient_adpt_3, data_time_transient_adpt_6, data_time_transient_adpt_9, data_time_transient_kmns_3,
        data_time_transient_kmns_6, data_time_transient_kmns_9]
labels = ['ADPT-3', 'ADPT-6', 'ADPT-9', 'KMNS-3', 'KMNS-6', 'KMNS-9']


double_axis_boxplot(
    group_arrays=data,
    group_labels=labels,
    figsize=(6, 2.7),
    n_ticks=5,
    axis2_ylim=[0, 6],
    axis1_ylim=[40, 44],
    axis1_label=r"$\ell_\textup{cl}(1,12)$",
    axis2_label='Solver time [s]',
    showfliers=False
)

#%% CHECK
print("\n### CHECK: ADPT-3")
print(np.median(cost_avgT_adpt_3))

print("\n### CHECK: ADPT-6")
print(np.median(cost_avgT_adpt_6))

print("\n### CHECK: ADPT-9")
print(np.median(cost_avgT_adpt_9))

print("\n### CHECK: KMNS-3")
print(np.median(cost_avgT_kmns_3))

print("\n### CHECK: KMNS-6")
print(np.median(cost_avgT_kmns_6))

print("\n### CHECK: KMNS-9")
print(np.median(cost_avgT_kmns_9))




#%% Tightening
time_tightening_adpt_3 = adpt_3_all_vars['time_tightening_matrix'][:, 0:10].mean()
time_tightening_adpt_6 = adpt_6_all_vars['time_tightening_matrix'][:, 0:10].mean()
time_tightening_adpt_9 = adpt_9_all_vars['time_tightening_matrix'][:, 0:10].mean()

time_tightening_kmns_3 = kmns_3_all_vars['time_tightening_matrix'].mean()
time_tightening_kmns_6 = kmns_6_all_vars['time_tightening_matrix'].mean()
time_tightening_kmns_9 = kmns_9_all_vars['time_tightening_matrix'].mean()

print('time_tightening_adpt_3', time_tightening_adpt_3)
print('time_tightening_adpt_6', time_tightening_adpt_6)
print('time_tightening_adpt_9', time_tightening_adpt_9)

print('time_tightening_kmns_3', time_tightening_kmns_3)
print('time_tightening_kmns_6', time_tightening_kmns_6)
print('time_tightening_kmns_9', time_tightening_kmns_9)