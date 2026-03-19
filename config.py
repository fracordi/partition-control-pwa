import numpy as np

#%% Define variables for closed loop simulations

# Ch. constraint params
eps_nom = 0.15
delta = 0.05
beta = 1e-4

# Sys. params
n_x = 3
n_u = 1
n_uncertainty = 2
norm_type = 1
N_horizon = 5
N_control = 5

# Constraints
x2_low, x2_high = -1, 8
x3_low, x3_high = -1, 5

Hx = np.array([[0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
hx = np.array([[-x2_low], [x2_high], [-x3_low], [x3_high]])
Hu = np.array([[1], [-1]])
hu = np.array([[1], [1]]) * 2
X, U = {'A': Hx, 'b': hx}, {'A': Hu, 'b': hu}

# Cost
Qweight, Rweight = 5, 1
Q = np.eye(n_x) * Qweight
R = np.eye(n_u) * Rweight
x_ref = np.ones((n_x, 1)) * 0
u_ref = np.ones((n_u, 1)) * 0

# Dynamics
a1, a2, a3 = .5, .8, .9
A = [np.array([[0.6, 1, 1],
               [0, a1, 1.1],
               [0.1, 0, 0.4]]),
     np.array([[0.6, 1, 1],
               [0, a2, 1.1],
               [0.1, 0, 0.4]]),
     np.array([[0.6, 1, 1],
               [0, a3, 1.1],
               [0.1, 0, 0.4]])
     ]
for A_dyn in A:
    print(np.linalg.eig(A_dyn)[0])

B = np.array([[0.5], [0], [1]])
C = np.array([[0, 0], [1, 0], [0, 1]])
e1 = -0.5
e2 = 0.5
v = [np.array([[0], [e1 * (a2 - a1)], [0]]),
     np.zeros((n_x, 1)),
     np.array([[0], [e2 * (a2 - a3)], [0]])]
E1, E2 = 1, 0  # np.random.uniform(0,1), np.random.uniform(0,1)
E = [np.array([[0, E1, 0]]), np.array([[0, -E1, 0], [0, E1, 0]]), np.array([[0, -E1, 0]])]
e = [np.array([[e1]]), np.array([[-e1], [e2]]), np.array([[-e2]])]
mode_1 = {'A': A[0], 'B': B, 'C': C, 'v': v[0], 'E': E[0], 'e': e[0]}
mode_2 = {'A': A[1], 'B': B, 'C': C, 'v': v[1], 'E': E[1], 'e': e[1]}
mode_3 = {'A': A[2], 'B': B, 'C': C, 'v': v[2], 'E': E[2], 'e': e[2]}
modes = [mode_1, mode_2, mode_3]

stab_threshold = 0.3

T_cl = 30

x0 = np.array([[5], [5], [3]])
SYS_PARAMS = {'n_x': n_x, 'n_u': n_u, 'n_uncertainty': n_uncertainty, 'eps': eps_nom, 'delta': delta, 'beta': beta,
              'norm_type': norm_type, 'N_horizon': N_horizon, 'N_control': N_control, 'T_cl': T_cl,
            'eps_nom': eps_nom, 'A': A, 'B': B, 'C': C, 'v': v, 'E': E, 'e': e, 'modes': modes,
            'Hx': Hx, 'hx': hx, 'Hu': Hu, 'hu': hu, 'X': X, 'U': U, 'Q': Q, 'R': R, 'x_ref': x_ref, 'u_ref': u_ref,
              'x0': x0, 'stab_threshold': stab_threshold}


#%% Truncated Gaussian parameters
trc_mean = np.array([0, 0])
trc_stdev = np.array([0.06, 0.04])
intervals = np.ndarray((n_uncertainty, 2))

intervals[0, 0] = -0.12
intervals[0, 1] = 0.12
intervals[1, 0] = -0.1
intervals[1, 1] = 0.1
TRCGAUSS_PARAMS = {'type': 'TRUNC_GAUSSIAN', 'n_uncertainty': n_uncertainty, 'mean_vec': trc_mean, 'std_vec': trc_stdev, 'intervals': intervals}


#%% For numerical simulations

def get_constraints_params(n_x):
    x_bound = 1
    y_bound = 3
    Hx = np.vstack((np.eye(n_x), -np.eye(n_x)))
    hx = np.ones((2 * n_x,)) * x_bound
    Hy = np.array([-1, 1])
    hy = np.array([1, 1]) * y_bound
    X = {'A': Hx, 'b': hx}
    Y = {'A': Hy, 'b': hy}
    return x_bound, y_bound, X, Y


def parametric_sinusoid_params():
    """Parameters for parametric sinusoid (st. dev. relevant only if distribution is Gaussian,
        otherwise it should be uniform"""
    a_min = 0.3
    a_max = 0.6
    a_stdev = 0.05
    omega_min, omega_max, omega_std = 0.09, 0.18, 0.
    return a_min, a_max, a_stdev, omega_min, omega_max, omega_std

