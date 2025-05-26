import numpy as np

##% Define the system dynamics
n_x = 3
n_u = 1
n_uncertainty = 2
norm_type = 1
N_horizon = 5
eps_nom = 0.15

A = [np.array([[0.8, 1, 1],
               [0, 0.9, 1],
              [0, 0, 0.2]]),
     np.array([[0.8, 1, 1],
               [0, -0.9, 1],
              [0, 0, -0.2]]),
     np.array([[0.8, 1, 1],
               [0, 0.5, 1],
               [0, 0, 0.5]])
     ]
B = np.array([[0], [0], [1]])
C = np.array([[0, 0], [1, 0], [0, 1]])
v = np.zeros((n_x, 1))

# For PWA only
E = [np.array([[-1, 0, 0]]), np.array([[1, 0, 0], [-1, 0, 0]]), np.array([[1, 0, 0]])]
e = [np.array([[-1]]), np.array([[1], [1]]), np.array([[-1]])]
mode_1 = {'A': A[0], 'B': B, 'C': C, 'v': v, 'E': E[0], 'e': e[0]}
mode_2 = {'A': A[1], 'B': B, 'C': C, 'v': v, 'E': E[1], 'e': e[1]}
mode_3 = {'A': A[2], 'B': B, 'C': C, 'v': v, 'E': E[2], 'e': e[2]}
modes = [mode_1, mode_2, mode_3]

# Cost and constraints
x_bound = 0.7
u_bound = 0.7
Hx = np.array([[0, 0, -1], [0, 0, 1]])
hx = np.array([[1], [1]]) * x_bound
Hu = np.array([[1], [-1]])
hu = np.array([[1], [1]]) * u_bound
Qweight, Rweight = 2, 1
Q = np.eye(n_x) * Qweight
R = np.eye(n_u) * Rweight
x_ref = np.ones((n_x, 1)) * 0
u_ref = np.ones((n_u, 1)) * 0
x0 = np.array([[1.5], [2], [1]])

SYS_PARAMS = {'n_x': n_x, 'n_u': n_u, 'n_uncertainty': n_uncertainty, 'norm_type': norm_type, 'N_horizon': N_horizon,
            'eps_nom': eps_nom, 'modes': modes,
            'Hx': Hx, 'hx': hx, 'Hu': Hu, 'hu': hu, 'Q': Q, 'R': R, 'x_ref': x_ref, 'u_ref': u_ref, 'x0': x0}


#%% Sinusoidal uncertainty parameters
a_min = np.array([0.02, 0.02]) * np.array([1, 2])
a_max = np.array([0.03, 0.03]) * np.array([1, 2])
omega_means = np.array([[0.05, 0.12, 0.3, 0.5, 0.75], [0.05, 0.12, 0.3, 0.5, 0.75]]) * np.array([[2], [1]])
omega_weights = np.array([[0.05, 0.1, 0.4, 0.4, 0.05], [0.05, 0.1, 0.4, 0.4, 0.05]])
omega_std = np.array([0.01, 0.01]) * np.array([1, 2])
p_min = np.array([-0.1, -0.1])
p_max = np.array([0.1, 0.1])
w_min = np.array([-0.03, -0.03]) * np.array([1, 1])
w_max = np.array([0.03, 0.03]) * np.array([1, 1])

SINUSOIDAL_PARAMS = {'type': 'SINUSOIDAL', 'n_uncertainty': n_uncertainty, 'a_min': a_min, 'a_max': a_max,
                     'omega_means': omega_means, 'omega_weights': omega_weights, 'omega_std': omega_std, 'p_min': p_min,
                     'p_max': p_max, 'w_min': w_min, 'w_max': w_max}

#%% Mixture uncertainty parameters
n_components = 3
components_mean = np.array([np.linspace(-0.08, 0.08, n_components), np.linspace(-0.08, 0.08, n_components)]) \
                  * np.array([[1], [0.9]])
weights = np.array([[1, 4, 1], [1, 4, 1]])
weights = weights / np.sum(weights, axis=1).reshape(-1, 1)
stdev = np.array([[0.01 for _ in range(n_components)], [0.01 for _ in range(n_components)]]) \
                  * np.array([[1], [0.9]])
MIXTURE_PARAMS = {'type': 'MIXTURE', 'n_uncertainty': n_uncertainty, 'n_components': n_components,
                  'components_mean': components_mean, 'components_weights': weights, 'components_stdev': stdev}

#%%
UNC_PARAMS = {'sinusoidal': SINUSOIDAL_PARAMS, 'mixture': MIXTURE_PARAMS}


