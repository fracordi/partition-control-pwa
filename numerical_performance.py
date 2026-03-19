from stochastic_optimization import PartitionOptimizerLP
import numpy as np
from utils import check_probability_sum, compute_stats, solve_binomial_sum, get_random_matrices
from config import parametric_sinusoid_params, get_constraints_params
from uncertainty_utils import ParametricSinusoid
import math
import time
import pickle
import os

#%% Main parameters
SAVE = False
TYPE = 'partition'
n_x = 10
n_constr = n_x
n_uncertainty = 10
epsilon = 0.15
delta = 0.05
beta = 1e-5

N_exp_confidence = 500
M_grid_list = [[2, 2], [4, 4], [6, 6], [8, 8], [10, 10], [14, 14]]
params_distribution = 'uniform'
norm_type = 1
N_samples_test = 100000

print(f'### Simulation: {TYPE}, n_x: {n_x}, n_uncertainty: {n_uncertainty}, params_distribution: '
      f'{params_distribution}, delta: {delta}, norm type: {norm_type}, num exp. confidence: {N_exp_confidence} '
      f'M grid list: {M_grid_list}, Save: {SAVE} ###')

num_instances = 1
n_params = 2        # for sinusoid: a, \omega
seed_matrix_instances = 1
seed_samples_batches = 2
seed_empirical = 3

SOLVER = 'GUROBI'
SIMPLIFY = True
num_modes = 1
partition_strategy = 'full_grid'    # 'clustering' # 'iterative_gird'
K_list = [math.prod(M_grid) for M_grid in M_grid_list]

optimizer = PartitionOptimizerLP(n_x, n_uncertainty, SOLVER)
x_bound, y_bound, X, Y = get_constraints_params(n_x)

#%% Uncertainty
a_min, a_max, a_stdev, omega_min, omega_max, omega_std = parametric_sinusoid_params()
generator = ParametricSinusoid(n_uncertainty, a_min, a_max, a_stdev, omega_min, omega_max, omega_std, params_distribution)
diam_unc = 2 * n_uncertainty * a_max
diam_x = np.sqrt((2*x_bound)**2 * n_x + (2*y_bound)**2)
r = 0.01

#%% Test
D1_all, D2_all, d_all, E1_all, E2_all, E3_all, e_all, e_feas = get_random_matrices(seed_matrix_instances, num_instances,
                                                                           num_modes, n_constr,
                                                                           n_x, n_uncertainty, uncertainty_in_cost=True,
                                                                           ensure_feasibility=True)
D1, D2, d, E1, E2, E3, e = D1_all[0][0], D2_all[0][0], d_all[0][0], E1_all[0], E2_all[0], E3_all[0], e_all[0]

store_cost_pp = np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_cost_rp = np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_cost_lb_th = np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_cost_lb_aux = np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_cost_ub = np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_empirical_cost = np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_empirical_violation = np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_solver_time_pp = np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_compile_time_pp = np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_solver_time_rp = np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_compile_time_rp = np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_tight_time = np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_relax_time = np.full((len(K_list), N_exp_confidence), -1, dtype=float)

store_c1= np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_c2= np.full((len(K_list), N_exp_confidence), -1, dtype=float)
store_c3= np.full((len(K_list), N_exp_confidence), -1, dtype=float)

samples_test = generator.get_samples(N_samples_test, seed_empirical)

for k, M_grid in enumerate(M_grid_list):

    print(f'\n\n### M grid = {M_grid} (num. conf. exp: {N_exp_confidence}) ###')
    regions_params, boxes_params = generator.grid_params(M_grid)
    K = len(boxes_params)
    N_samples = int((K * np.log(2) + np.log(1 / beta)) / (2 * delta ** 2)) + 1
    print('N samples =', N_samples)
    samples_params_batches = generator.sample_params(N_samples, seed=seed_samples_batches, n_batches=N_exp_confidence)

    for i in range(N_exp_confidence):
        # Extract samples for i-th simulation
        samples = samples_params_batches[i]
        p_hat, nominal_scen_params, regions_new_params, boxes_new_params, clusters_params = generator.compute_partition_elements(
            samples, regions_params, boxes_params, N_samples, traslate=False)
        check_probability_sum(p_hat)
        nominal_scen = generator.from_params_to_sin(nominal_scen_params)

        # CHECK THIS
        clusters = [generator.from_params_to_sin(cluster_params) for cluster_params in clusters_params]

        regions_new, boxes_new = generator.partition_from_params(boxes_new_params, nominal_scen, traslate=True)

        time_tight_start = time.time()
        optimizer.set_tightening(D2, boxes=boxes_new)
        time_tight_end = time.time()
        optimizer.set_gamma(D2, boxes=boxes_new)
        time_rel_end = time.time()

        (cost_pp, cost_rp, cost_lb_th, cost_lb_aux, cost_ub, empirical_cost, empirical_violation, solver_time_pp, compile_time_pp,
         solver_time_rp, compile_time_rp, c1, c2, c3) = (optimizer.get_performances(X, Y, D1, D2, d, E1, E2, E3, e, e_feas, nominal_scen,
                                                                   clusters, p_hat, epsilon, delta, norm_type, N_samples,
                                                                   diam_unc, diam_x, r, beta, samples_test))

        # If feasible store data (problem is always feasible if ensure_feasibility=True, through variable y)
        if cost_pp is not None:
            store_cost_pp[k, i] = cost_pp
            store_cost_rp[k, i] = cost_rp
            store_cost_lb_th[k, i] = cost_lb_th
            store_cost_lb_aux[k, i] = cost_lb_aux
            store_cost_ub[k, i] = cost_ub
            store_empirical_cost[k, i] = empirical_cost
            store_empirical_violation[k, i] = empirical_violation
            store_solver_time_pp[k, i] = solver_time_pp
            store_compile_time_pp[k, i] = compile_time_pp
            store_solver_time_rp[k, i] = solver_time_rp
            store_compile_time_rp[k, i] = compile_time_rp
            store_tight_time[k, i] = time_tight_end - time_tight_start
            store_relax_time[k, i] = time_rel_end - time_tight_end
            store_c1[k, i] = c1
            store_c2[k, i] = c2
            store_c3[k, i] = c3

#%% Results
stats_cost_pp = compute_stats(store_cost_pp, K_list)
stats_cost_rp = compute_stats(store_cost_rp, K_list)
stats_cost_lb_th = compute_stats(store_cost_lb_th, K_list)
stats_cost_lb_aux = compute_stats(store_cost_lb_aux, K_list)
stats_cost_ub = compute_stats(store_cost_ub, K_list)
stats_empirical_cost = compute_stats(store_empirical_cost, K_list)
stats_empirical_violation = compute_stats(store_empirical_violation, K_list)
stats_solver_time_pp = compute_stats(store_solver_time_pp, K_list)
stats_compile_time_pp = compute_stats(store_compile_time_pp, K_list)
stats_solver_time_rp = compute_stats(store_solver_time_rp, K_list)
stats_compile_time_rp = compute_stats(store_compile_time_rp, K_list)
stats_tight_time = compute_stats(store_tight_time, K_list)
stats_rel_time = compute_stats(store_relax_time, K_list)
stats_c1 = compute_stats(store_c1, K_list)
stats_c2 = compute_stats(store_c2, K_list)
stats_c3 = compute_stats(store_c3, K_list)

print(f'\n\n### Results ###')
print('\nCost pp\n', stats_cost_pp)
print('\nCost rp\n', stats_cost_rp)
print('\nCost lb th\n', stats_cost_lb_th)
print('\nCost lb aux\n', stats_cost_lb_aux)
print('\nCost ub\n', stats_cost_ub)
print('\nJ_real\n', stats_empirical_cost)

print('\nEmpirical violation\n', stats_empirical_violation)

print('\nSolver time pp\n', stats_solver_time_pp)
print('\nCompile time pp\n', stats_compile_time_pp)
print('\nSolver time rp\n', stats_solver_time_rp)
print('\nCompile time rp\n', stats_compile_time_rp)

print('\nTight time rp\n', stats_tight_time)
print('\nRelax time rp\n', stats_rel_time)

print('\nC1\n', stats_c1)
print('\nC2\n', stats_c2)
print('\nC3\n', stats_c3)

#%% Save
SIMULATION = 'numeric_performance_'
if SAVE:
    variables_to_save = {'sim': 'numeric_sin', 'type': TYPE, 'n_x': n_x, 'n_unc': n_uncertainty, 'M_grid_list': M_grid_list,
                         'norm_type': norm_type, 'params_distribution': params_distribution, 'N_exp_confidence': N_exp_confidence,
                         'num_modes': num_modes, 'eps':epsilon, 'delta':delta,
                         'beta':beta, 'X': X, 'Y': Y, 'cost_pp': store_cost_pp, 'cost_rp': store_cost_rp, 'store_cost_lb_th':
                        store_cost_lb_th, 'store_cost_lb_aux': store_cost_lb_aux, 'store_cost_ub': store_cost_ub,
                         'store_empirical_cost': store_empirical_cost, 'store_empirical_violation': store_empirical_violation,
                         'store_solver_time_pp': store_solver_time_pp, 'store_compile_time_pp': store_compile_time_pp,
                         'store_solver_time_rp': store_solver_time_rp, 'store_compile_time_rp': store_compile_time_rp,
                         'store_tight_time': store_tight_time, 'store_relax_time': store_relax_time, 'c1': store_c1,
                         'c2': store_c2, 'c3': store_c3
                         }
    # Save variables to a file
    file_name = (SIMULATION + str(TYPE) + '_nx' + str(n_x) + '_nunc' + str(n_uncertainty)
                 + '_N_exp_confidence' + str(N_exp_confidence) + '_delta' + str(delta) + '.pkl')
    file_name = os.path.join('variables_numerical', file_name)
    with open(file_name, 'wb') as file:
        pickle.dump(variables_to_save, file)

