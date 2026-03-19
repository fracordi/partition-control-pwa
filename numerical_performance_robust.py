from stochastic_optimization import RobOptimizerLP
import numpy as np
import polytope as pt
from utils import compute_stats, solve_binomial_sum, get_random_matrices, solve_robust
from config import parametric_sinusoid_params, get_constraints_params
from uncertainty_utils import ParametricSinusoid
import pickle
import os

#%% Main parameters
SAVE = False
TYPE = 'robust'
n_x = 10
n_constr = n_x
n_uncertainty = 10
epsilon = 0.15
beta = 1e-5
cost_type = 'avg'

N_exp_confidence = 500
params_distribution = 'uniform'
norm_type = 1
N_samples_test = 100000

print(f'### Simulation: {TYPE}, n_x: {n_x}, n_uncertainty: {n_uncertainty}, params_distribution: '
      f'{params_distribution}, norm type: {norm_type}, num exp. confidence: {N_exp_confidence} ###')

num_instances = 1
n_params = 2        # for sinusoid: a, \omega
seed_matrix_instances = 1
seed_samples_batches = 2
seed_empirical = 3

SOLVER = 'GUROBI'
SIMPLIFY = True
num_modes = 1

#%% Uncertainty
# Construct box that contains fraction 1-eps of uncertainty: num decision variables is 2 * n_uncertainty
N_samples = solve_binomial_sum(2 * n_uncertainty, epsilon, beta, num_modes)
optimizer = RobOptimizerLP(n_x, n_uncertainty, SOLVER)
x_bound, y_bound, X, Y = get_constraints_params(n_x)
a_min, a_max, a_stdev, omega_min, omega_max, omega_std = parametric_sinusoid_params()
generator = ParametricSinusoid(n_uncertainty, a_min, a_max, a_stdev, omega_min, omega_max, omega_std, params_distribution)

#%% Test
D1_all, D2_all, d_all, E1_all, E2_all, E3_all, e_all, e_feas = get_random_matrices(seed_matrix_instances, num_instances,
                                                                           num_modes, n_constr,
                                                                           n_x, n_uncertainty, uncertainty_in_cost=True,
                                                                           ensure_feasibility=True)
D1, D2, d, E1, E2, E3, e = D1_all[0][0], D2_all[0][0], d_all[0][0], E1_all[0], E2_all[0], E3_all[0], e_all[0]

store_cost = np.full(N_exp_confidence, -1, dtype=float)
store_empirical_cost = np.full(N_exp_confidence, -1, dtype=float)
store_empirical_violation = np.full(N_exp_confidence, -1, dtype=float)
store_solver_time = np.full(N_exp_confidence, -1, dtype=float)
store_compile_time = np.full(N_exp_confidence, -1, dtype=float)

samples_test = generator.get_samples(N_samples_test, seed_empirical)

print('N samples =', N_samples)
samples_params_batches = generator.sample_params(N_samples, seed=seed_samples_batches, n_batches=N_exp_confidence)

for i in range(N_exp_confidence):
    # Extract samples for i-th simulation
    samples_param = samples_params_batches[i]
    box_params = generator.bounding_box(samples_param)
    box_poly = pt.box2poly(box_params)
    region_params = {'A': box_poly.A, 'b': box_poly.b.reshape(-1, 1)}

    cost, x, y, solver_time, compile_time, status = solve_robust(X, Y, D1, D2, d, E1, E2, E3, e, e_feas, norm_type,
                                                                     optimizer, generator, samples_param, [region_params],
                                                                     [box_params], N_samples,
                                                                     cost_type=cost_type)

    # If feasible store data (problem is always feasible if ensure_feasibility=True, through variable y)
    if cost is not None:
        empirical_cost, empirical_violation = optimizer.compute_empirical_metrics(x, y, samples_test, D1, D2, d,
                                                                                  E1,
                                                                                  E2, E3, e, e_feas,
                                                                                  norm_type=norm_type)
        store_cost[i] = cost
        store_empirical_cost[i] = empirical_cost
        store_empirical_violation[i] = empirical_violation
        store_solver_time[i] = solver_time
        store_compile_time[i] = compile_time

#%% Results
flatten = True
stats_cost = compute_stats(store_cost, flatten=flatten)
stats_empirical_cost = compute_stats(store_empirical_cost, flatten=flatten)
stats_violation = compute_stats(store_empirical_violation, flatten=flatten)
stats_solver_time = compute_stats(store_solver_time, flatten=flatten)
stats_compile_time = compute_stats(store_compile_time, flatten=flatten)

print(f'\n\n### Results ###')
print('\nCost\n', stats_cost)
print('\nJ_real\n', stats_empirical_cost)

print('\nEmpirical violation\n', stats_violation)

print('\nSolver time\n', stats_solver_time)
print('\nCompile time\n', stats_compile_time)

#%% Save
SIMULATION = 'numeric_'
if SAVE:
    variables_to_save = {'sim': 'numeric_sin', 'type': TYPE, 'n_x': n_x, 'n_unc': n_uncertainty,
                         'N_exp_confidence': N_exp_confidence,'norm_type': norm_type, 'cost_type': cost_type,
                         'N_samples': N_samples, 'params_distribution': params_distribution, 'num_modes': num_modes, 'eps':epsilon,
                         'beta':beta, 'X': X, 'Y': Y, 'store_cost': store_cost, 'store_empirical_cost': store_empirical_cost,
                         'store_empirical_violation': store_empirical_violation, 'store_solver_time': store_solver_time,
                         'store_compile_time': store_compile_time}
    # Save variables to a file
    file_name = (SIMULATION + str(TYPE) + '_nx' + str(n_x) + '_nunc' + str(n_uncertainty) + '_eps' + str(epsilon)
                 +'_Nsamples' + str(N_samples) + '_N_exp_confidence' + str(N_exp_confidence) + '_cost_' + str(cost_type)
                 + '.pkl')
    file_name = os.path.join('variables_numerical', file_name)
    with open(file_name, 'wb') as file:
        pickle.dump(variables_to_save, file)

