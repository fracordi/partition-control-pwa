from stochastic_optimization import PartitionOptimizerLP, RandomizedOptimizer
import numpy as np
from utils import compute_stats, solve_binomial_sum, get_random_matrices, solve_pp
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
n_uncertainty = 10
n_constr = n_x
norm_type = 1
params_distribution = 'uniform'
num_instances = 100

print(f'### Simulation: {TYPE}, n_x: {n_x}, n_uncertainty: {n_uncertainty}, params_distribution: '
      f'{params_distribution}, norm type: {norm_type}, num_instances: {num_instances} ###')

N_exp_confidence = 1
M_grid = [10, 5]
epsilon = 0.15
delta = 0.05
beta = 1e-5
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
    print(f'Gridding: {M_grid}')

N_samples = int((K * np.log(2) + np.log(1 / beta)) / (2 * delta ** 2)) + 1 if TYPE == 'partition' \
    else solve_binomial_sum(n_x+1, epsilon, beta, num_modes)

optimizer = PartitionOptimizerLP(n_x, n_uncertainty, SOLVER) if TYPE == 'partition' \
    else RandomizedOptimizer(n_x, n_uncertainty, SOLVER)

x_bound, y_bound, X, Y = get_constraints_params(n_x)

#%% Uncertainty
a_min, a_max, a_stdev, omega_min, omega_max, omega_std = parametric_sinusoid_params()
generator = ParametricSinusoid(n_uncertainty, a_min, a_max, a_stdev, omega_min, omega_max, omega_std, params_distribution)
regions_params, boxes_params = generator.grid_params(M_grid)
K = len(boxes_params)

samples_params_batches = generator.sample_params(N_samples, seed=seed_samples_batches, n_batches=N_exp_confidence)

#%% Test
store_cost = np.full((num_instances, N_exp_confidence), -1, dtype=float)
store_empirical_cost = np.full((num_instances, N_exp_confidence), -1, dtype=float)
store_empirical_violation = np.full((num_instances, N_exp_confidence), -1, dtype=float)
store_solver_time = np.full((num_instances, N_exp_confidence), -1, dtype=float)
store_compile_time = np.full((num_instances, N_exp_confidence), -1, dtype=float)

samples_test = generator.get_samples(N_samples_test, seed_empirical)

D1_all, D2_all, d_all, E1_all, E2_all, E3_all, e_all, e_feas = get_random_matrices(seed_matrix_instances, num_instances,
                                                                           num_modes, n_constr,
                                                                           n_x, n_uncertainty, uncertainty_in_cost=True,
                                                                           ensure_feasibility=True)
infeasible = False
for k in range(num_instances):

    print(f'\n\n### Instance {k + 1} / {num_instances} ###')

    D1 = D1_all[k][0]
    D2 = D2_all[k][0]
    d = d_all[k][0]
    E1 = E1_all[k]
    E2 = E2_all[k]
    E3 = E3_all[k]
    e = e_all[k]

    for i in range(N_exp_confidence):

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

        elif TYPE== 'random':
            samples_random = generator.from_params_to_sin(samples_params)
            cost, x, y, solver_time, compile_time, status = optimizer.optimize(X, Y, D1, D2, d, E1, E2, E3, e, e_feas,
                                                                               samples_random, norm_type=norm_type)
            if cost is None:
                infeasible = True
        else:
            raise ValueError(f'Simulation must be either "partition" or "random".')

        # If feasible store data (problem is always feasible if ensure_feasibility=True, through variable y)
        if cost is not None:
            empirical_cost, empirical_violation = optimizer.compute_empirical_metrics(x, y, samples_test, D1, D2, d, E1,
                                                                                      E2, E3, e, e_feas,
                                                                                      norm_type=norm_type)
            store_cost[k, i] = cost
            store_empirical_cost[k, i] = empirical_cost
            store_empirical_violation[k, i] = empirical_violation
            store_solver_time[k, i] = solver_time
            store_compile_time[k, i] = compile_time

if infeasible:
    raise Warning('Infeasible instances detected.')

#%% Results
flatten = True if N_exp_confidence==1 else False
stats_cost = compute_stats(store_cost, flatten=flatten)
stats_emp_cost = compute_stats(store_empirical_cost, flatten=flatten)
stats_emp_viol = compute_stats(store_empirical_violation, flatten=flatten)
stats_solve_time = compute_stats(store_solver_time, flatten=flatten)
stats_compile_time = compute_stats(store_compile_time, flatten=flatten)

print(f'\n\n### Results PP (K = {K}, delta={delta}) ###') if TYPE == 'partition' else print(f'\n\n### Results RA ###')
print('\nCost\n', stats_cost)
print('\nEmpirical Cost\n', stats_emp_cost)
print('\nEmpirical Violation\n', stats_emp_viol)
print('\nSolver time\n', stats_solve_time)
print('\nCompile time\n', stats_compile_time)

#%% Save
SIMULATION = 'numeric_'
if SAVE:
    variables_to_save = {'sim': 'numeric', 'type': TYPE, 'n_x': n_x, 'n_unc': n_uncertainty,
                         'M_grid': M_grid, 'norm_type': norm_type, 'uncertainty_in_cost': True,
                         'ensure_feasibility': True, 'seed_samples_batches': seed_samples_batches,
                         'seed_matrix_instances': seed_matrix_instances, 'seed_empirical': seed_empirical,
                         'params_distribution': params_distribution,
                         'num_modes': num_modes, 'constraint_modification': 'tight',
                         'eps':epsilon, 'delta':delta, 'beta':beta, 'N_samples': N_samples, 'X': X, 'Y': Y,
                         'cost': store_cost,
                         'empirical_cost': store_empirical_cost, 'empirical_violation': store_empirical_violation,
                         'solver_time': store_solver_time, 'compile_time': store_compile_time,
                         }
    # Save variables to a file
    if TYPE=='partition':
        file_name = SIMULATION + str(TYPE) + '_nx' + str(n_x) + '_nunc' + str(n_uncertainty) + '_K' + str(K) + '.pkl'
    else:
        file_name = SIMULATION + str(TYPE) + '_nx' + str(n_x) + '_nunc' + str(n_uncertainty) + '.pkl'
    file_name = os.path.join('variables_numerical', file_name)
    with open(file_name, 'wb') as file:
        pickle.dump(variables_to_save, file)

