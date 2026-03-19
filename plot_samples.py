import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import matplotlib as mpl
import matplotlib.ticker as mticker

sns.set_theme()
TINY_SIZE = 4
SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 13
rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('legend', fontsize=TINY_SIZE)  # legend fontsize
rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
# plt.rc["font"] = "serif"
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))



#%% Use formulas to compute number of samples and function for which equality holds
beta = 10**(-4)
N = lambda K, delta: np.ceil(((K*np.log(2) + np.log(1/beta))/ (2 * delta**2)))
Ncvx = lambda eps, d, n_bin: np.ceil(2/eps * (d + n_bin * np.log(2 / beta)))
Nequal = lambda d, K, n_bin: (K*np.log(2) + np.log(1/beta)) / (d + n_bin * np.log(2 / beta))

# plt.rc('font', family='serif')
mrk_size = 9

#%% 3D plot sample complexity PP
K_range = np.linspace(1, 200, 200)
delta_range = np.linspace(0.02, 0.05, 100)
K_mesh, delta_mesh = np.meshgrid(K_range, delta_range)
Z = N(K_mesh, delta_mesh)   # no transpose

fig = plt.figure(figsize=(3.8, 2.), dpi=150)
ax = fig.add_subplot(111, projection="3d")

vmax = np.nanmax(np.abs(Z))
vmin = 0

surf = ax.plot_surface(K_mesh, delta_mesh, Z, cmap="RdBu_r", vmin=vmin, vmax=vmax, linewidth=0, antialiased=True,)
ax.set_yticks([0.01, 0.02, 0.03, 0.04, 0.05])
ax.set_xlabel(r"$K$")
ax.set_ylabel(r"$\delta$")
ax.set_zlabel(r"$N$")

ax.set_xlim(K_range.min(), K_range.max())
ax.set_ylim(delta_range.min(), delta_range.max())
ax.set_zlim(vmin, vmax)
ax.set_zticks([50000, 100000, 150000, 200000])


ax.view_init(elev=28, azim=45)

# White background
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
#ax.grid(False)

# remove fill
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
ax.grid(True)

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor((1, 1, 1, 0))
    axis.pane.set_edgecolor("lightgray")

ax.grid(True)

ax.xaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.6)
ax.yaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.6)
ax.zaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.6)

ax.xaxis._axinfo["grid"]["linewidth"] = 0.6
ax.yaxis._axinfo["grid"]["linewidth"] = 0.6
ax.zaxis._axinfo["grid"]["linewidth"] = 0.6

plt.tight_layout()
plt.show()


#%% Colorbars for N - N_rnd

d = np.linspace(100, 2500, 200)
eps = np.linspace(0.01, 0.1, 100)

N_binaries_1 = 1
N_binaries_2 = 10

D, E = np.meshgrid(d, eps)

# N partitioning and N_random with 1 and 2^10 subproblems
N_partitioning = N(K=10, delta=eps/2).reshape(-1, 1)
N_sa1 = Ncvx(E, D, N_binaries_1)
Z1 = N_partitioning - N_sa1
N_sa2 = Ncvx(E, D, N_binaries_2)
Z2 = N_partitioning - N_sa2

# Equality
d_curve = np.linspace(d.min(), d.max(), 400)
equality_line1 = Nequal(d_curve, K=10, n_bin=N_binaries_1)
equality_line2 = Nequal(d_curve, K=10, n_bin=N_binaries_2)

fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.6), dpi=150, constrained_layout=True)

Z_len = max(-min(Z1.min(), Z2.min()), max(Z1.max(), Z2.max()))

norm = mpl.colors.Normalize(
    vmin=-1*Z_len,
    vmax=1*Z_len
)

ticks = [-2.5e5, -1.2e5, 0, 1.2e5, 2.5e5]

im1 = ax[0].pcolormesh(D, E, Z1, shading="auto", cmap="RdBu_r", norm=norm)
ax[0].plot(d_curve, equality_line1, color="k", lw=1.2)
ax[0].set_xlabel(r"$n_x$")
ax[0].set_ylabel(r"$\varepsilon$")
ax[0].set_xticks([100, 1300, 2500])
ax[0].set_title(r"$Z=1$")


im2 = ax[1].pcolormesh(D, E, Z2, shading="auto", cmap="RdBu_r", norm=norm)
ax[1].plot(d_curve, equality_line2, color="k", lw=1.2)
ax[1].set_xlabel(r"$n_x$")
ax[1].set_xticks([100, 1300, 2500])
ax[1].set_title(r"$Z=2^{10}$")

cbar = fig.colorbar(im1, ax=ax)
cbar.set_ticks(ticks)
cbar.formatter = formatter
cbar.update_ticks()
cbar.set_label(r"$N-N_\textup{rnd}$")

ax[0].set_xlim(d.min(), d.max())
ax[0].set_ylim(eps.min(), eps.max())
ax[1].set_xlim(d.min(), d.max())
ax[1].set_ylim(eps.min(), eps.max())

# plt.tight_layout()
plt.show()


#%% Example: check equality line 1
for idx in range(len(equality_line1)):
    eps_test = equality_line1[idx]
    d_test = d_curve[idx]
    print('\n')
    print(N(K=10, delta=eps_test/2))
    print(Ncvx(eps_test, d_test, N_binaries_1))