from stochastic_optimization import PartitionOptimizerLP, RandomizedOptimizer
import numpy as np
from utils import compute_stats, get_random_matrices, solve_pp
from config import parametric_sinusoid_params, get_constraints_params
from uncertainty_utils import ParametricSinusoid
import math
import pickle
import os

#%% Main parameters
SAVE = False
TYPE = 'partition' # or 'random'
# TYPE = 'random'
n_x = 10
n_constr = n_x
n_uncertainty = 10
epsilon = 0.15
delta = 0.1
M_grid = [5, 5]
N_samples_list = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
N_exp_confidence = 500
norm_type = 1
params_distribution = 'uniform'
print_frequency = 50

print(f'### Simulation: {TYPE}, n_x: {n_x}, n_uncertainty: {n_uncertainty}, params_distribution: '
      f'{params_distribution}, norm type: {norm_type}, num exp. confidence: {N_exp_confidence}, save: {SAVE} ###')

num_instances = 1
N_samples_test = 100000
n_params = 2        # for sinusoid: a, \omega
seed_matrix_instances = 1
seed_samples_batches = 2
seed_empirical = 3

SOLVER = 'GUROBI'
num_modes = 1
partition_strategy = 'full_grid'    # 'clustering' # 'iterative_gird'
K = math.prod(M_grid)

if TYPE == 'partition':
    print(f'Gridding: {M_grid}, delta={delta}')

optimizer = PartitionOptimizerLP(n_x, n_uncertainty, SOLVER) if TYPE == 'partition' \
    else RandomizedOptimizer(n_x, n_uncertainty, SOLVER)

x_bound, y_bound, X, Y = get_constraints_params(n_x)

#%% Uncertainty
a_min, a_max, a_stdev, omega_min, omega_max, omega_std = parametric_sinusoid_params()
generator = ParametricSinusoid(n_uncertainty, a_min, a_max, a_stdev, omega_min, omega_max, omega_std, params_distribution)
regions_params, boxes_params = generator.grid_params(M_grid)
K = len(boxes_params)


#%% Test
D1_all, D2_all, d_all, E1_all, E2_all, E3_all, e_all, e_feas = get_random_matrices(seed_matrix_instances, num_instances,
                                                                           num_modes, n_constr,
                                                                           n_x, n_uncertainty, uncertainty_in_cost=True,
                                                                           ensure_feasibility=True)
D1, D2, d, E1, E2, E3, e = D1_all[0][0], D2_all[0][0], d_all[0][0], E1_all[0], E2_all[0], E3_all[0], e_all[0]
store_empirical_cost = np.full((len(N_samples_list), N_exp_confidence), -1, dtype=float)
store_empirical_violation = np.full((len(N_samples_list), N_exp_confidence), -1, dtype=float)
store_solver_time = np.full((len(N_samples_list), N_exp_confidence), -1, dtype=float)
store_compile_time = np.full((len(N_samples_list), N_exp_confidence), -1, dtype=float)

samples_test = generator.get_samples(N_samples_test, seed_empirical)
infeasible = False

for k, N_samples in enumerate(N_samples_list):

    print(f'\n\n### N samples = {N_samples} (num. conf. exp: {N_exp_confidence}) ###')

    # For each choice for Nsamples, create 500 samples sets
    samples_params_batches = generator.sample_params(N_samples, seed=seed_samples_batches, n_batches=N_exp_confidence)

    for i in range(N_exp_confidence):
        if (i + 1) % print_frequency == 0:
            print(f'Experiment {i + 1} / {N_exp_confidence}')

        # Extract samples for i-th simulation
        samples_params = samples_params_batches[i]

        if TYPE== 'partition':
            cost, x, y, mu, solver_time, compile_time, status = solve_pp(X, Y, D1, D2, d, E1, E2, E3, e,
                                                                         e_feas, epsilon, delta, norm_type, optimizer,
                                                                         'tight', generator,
                                                                         samples_params, regions_params, boxes_params,
                                                                         N_samples, partition_strategy)
            if cost is None:
                infeasible = True
        elif TYPE == 'random':
            samples_random = generator.from_params_to_sin(samples_params)
            cost, x, y, solver_time, compile_time, status = optimizer.optimize(X, Y, D1, D2, d, E1, E2, E3, e,
                                                                               e_feas,
                                                                               samples_random, norm_type=norm_type)
            if cost is None:
                infeasible = True
        else:
            raise ValueError(f'Simulation must be either "partition" or "random".')

        # If feasible store data (problem is always feasible if ensure_feasibility=True, through variable y)
        if cost is not None:
            empirical_cost, empirical_violation = optimizer.compute_empirical_metrics(x, y, samples_test, D1, D2, d,
                                                                                      E1,
                                                                                      E2, E3, e, e_feas,
                                                                                      norm_type=norm_type)
            store_empirical_cost[k, i] = empirical_cost
            store_empirical_violation[k, i] = empirical_violation
            store_solver_time[k, i] = solver_time
            store_compile_time[k, i] = compile_time
        else:
            raise ValueError('Choose either partition or random')

#%% Results
df_cost = compute_stats(store_empirical_cost, N_samples_list)
df_viol = compute_stats(store_empirical_violation, N_samples_list)
df_solve_time = compute_stats(store_solver_time, N_samples_list)
df_compile_time = compute_stats(store_compile_time, N_samples_list)

print(f'\n\n### Results PP (K = {K}, delta={delta}) ###') if TYPE == 'partition' else print(f'\n\n### Results RA ###')
print('\nEmpirical Cost\n', df_cost)
print('\nEmpirical Violation\n', df_viol)
print('\nSolver time\n', df_solve_time)
print('\nCompile time\n', df_compile_time)

#%% Save
SIMULATION = 'numeric_confidence_'
if SAVE:
    variables_to_save = {'sim': 'numeric', 'type': TYPE, 'n_x': n_x, 'n_unc': n_uncertainty,
                         'M_grid': M_grid, 'norm_type': norm_type, 'uncertainty_in_cost': True,
                         'ensure_feasibility': True, 'seed_samples_batches': seed_samples_batches,
                         'seed_matrix_instances': seed_matrix_instances, 'seed_empirical': seed_empirical,
                         'params_distribution': params_distribution,
                         'num_modes': num_modes, 'constraint_modification': 'tight',
                         'eps':epsilon, 'delta':delta, 'N_samples_list': N_samples_list, 'X': X, 'Y': Y,
                         'empirical_cost': store_empirical_cost, 'empirical_violation': store_empirical_violation,
                         'solver_time': store_solver_time, 'compile_time': store_compile_time,
                         }
    # Save variables to a file
    if TYPE=='partition':
        file_name = (SIMULATION + str(TYPE) + '_nx' + str(n_x) + '_nunc' + str(n_uncertainty) + '_K' + str(K)
                     + '_delta' + str(delta) + '_N_exp_confidence' + str(N_exp_confidence) + '.pkl')
    else:
        file_name = (SIMULATION + str(TYPE) + '_nx' + str(n_x) + '_nunc' + str(n_uncertainty)
                     + '_N_exp_confidence' + str(N_exp_confidence) +'.pkl')
    file_name = os.path.join('variables_numerical', file_name)
    with open(file_name, 'wb') as file:
        pickle.dump(variables_to_save, file)

