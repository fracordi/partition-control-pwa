import numpy as np
from config import SYS_PARAMS, SINUSOIDAL_PARAMS
from utils import filter_feasible_experiments, compute_stats, check_probability_sum
import polytope as pt
from controller import PartitionControllerPwa
from uncertainty import SinusoidalDisturbance
import pickle
import os
import time


#%% Params for simulations

SAVE = False
idx_experiment = 1
SOLVER = 'GUROBI'
N_samples_test = 100000
print_frequency = 10    # print every print_frequency experiments
N_exp_for_confidence = 125
K_list = [5, 10, 20, 50, 100, 200]
delta = 0.05
beta = 1e-4
idx_experiment = 1

store_lb_th = np.empty((len(K_list), N_exp_for_confidence))
store_lb_aux = np.empty((len(K_list), N_exp_for_confidence))
store_ub = np.empty((len(K_list), N_exp_for_confidence))
store_tot_time = np.empty((len(K_list), N_exp_for_confidence))
store_violations = np.empty((len(K_list), N_exp_for_confidence))
store_modification_time = np.empty((len(K_list), N_exp_for_confidence))

generator = SinusoidalDisturbance(SINUSOIDAL_PARAMS)
(n_x, n_u, n_uncertainty, norm_type, N_horizon, eps_nom, modes, Hx, hx, Hu, hu, Q, R, x_ref, u_ref, x0) = SYS_PARAMS.values()
X, U = {'A': Hx, 'b': hx}, {'A': Hu, 'b': hu}

# Get uncertainty domain and create initial polytope box.
lb, ub = generator.get_domain(N_horizon)
uncertainty_box = np.array([lb, ub]).transpose()
initial_box = pt.box2poly(uncertainty_box)
poly_u = pt.Polytope(U['A'], U['b'])

# Compute diameter for input polytope.
distances = [np.linalg.norm(u - v, 2) for u in pt.extreme(poly_u) for v in pt.extreme(poly_u)]
diam_U = max(distances) * np.sqrt(N_horizon)

# Set up the controller.
pp_controller = PartitionControllerPwa(n_x, n_u, n_uncertainty, N_horizon, modes,
                                       X, U, Q, R, x_ref, u_ref, norm_type, eps_nom, SOLVER)
pp_controller.set_compact_matrices(x0)
lip_u = pp_controller.set_lip_constant_u()
# lip_u = pp_controller.set_lip_constant_u_sampling(x0, uncertainty_box)
lip_unc = pp_controller.set_lip_constant_unc()
# lip_unc = pp_controller.set_lip_constant_unc_sampling(x0, uncertainty_box)
pp_controller.set_minimum_smallest_sv()
r = 1 / lip_u
diam_unc = np.linalg.norm(ub - lb, 1)

samples_test_violation = generator.get_samples(N_samples_test, N_horizon)
n_configs = len(K_list)

for i, K_ini in enumerate(K_list):
    regions, boxes = generator.box_partitioning(K_ini, lb, ub)
    N_samples = int((K_ini * np.log(2) + np.log(1 / beta)) / (2 * delta ** 2)) + 1

    print(f'\n\n### K = {K_ini} (Nsamples = {N_samples}) ###')

    lam = np.sqrt(((lip_unc * diam_unc) ** 2) / (2 * N_samples) *
                  (np.log(1 / beta) + n_u * N_horizon * np.log(3 * diam_U / r)))
    for h in range(N_exp_for_confidence):
        samples = generator.get_samples(N_samples, N_horizon)
        p_hat, nominal_scen, polytopes, regions_new, boxes_new, clusters = generator.compute_partition_elements(
            samples, regions, boxes, N_samples)
        check_probability_sum(p_hat)
        nominal_scen = np.array(nominal_scen)
        time_start_mod = time.time()
        pp_controller.set_tightening(nominal_scen, boxes=boxes_new)
        pp_controller.set_gamma(nominal_scen, boxes=boxes_new)
        store_modification_time[i, h] = time.time() - time_start_mod

        violation, cost_lb_th, cost_lb_aux, cost_ub, time_tot = pp_controller.get_performances(
            x0, lam, clusters, nominal_scen, p_hat, delta, N_samples,
            samples_test_violation, lb='both', r=r)

        store_violations[i, h] = violation
        store_lb_th[i, h] = cost_lb_th
        store_lb_aux[i, h] = cost_lb_aux
        store_ub[i, h] = cost_ub
        store_tot_time[i, h] = time_tot

        if (h + 1) % print_frequency == 0:
            print(f'\nExperiment {h + 1} / {N_exp_for_confidence}')
            print(f'Avg. Violation: {store_violations[i, :h + 1].mean()}')
            print(f'Min. LB (theoretical): {store_lb_th[i, :h + 1].min()}')
            print(f'Min. LB (auxiliary): {store_lb_aux[i, :h + 1].min()}')
            print(f'Max. UB: {store_ub[i, :h + 1].max()}')
            print(f'Avg. Time: {store_tot_time[i, :h + 1].mean()}')

# Filter out infeasible experiments.
(store_violations, store_lb_th, store_lb_aux, store_ub, store_tot_time) = filter_feasible_experiments(
    store_violations, [store_violations, store_lb_th, store_lb_aux, store_ub, store_tot_time]
)

df_viol = compute_stats(store_violations, K_list)
df_lb_th = compute_stats(store_lb_th, K_list)
df_lb_aux = compute_stats(store_lb_aux, K_list)
df_ub = compute_stats(store_ub, K_list)
df_time = compute_stats(store_tot_time, K_list)

print('\n\n ### Results ###')
print('\nViolation\n', df_viol)
print('\nLB - Theoretical\n', df_lb_th)
print('\nLB - Auxiliary\n', df_lb_aux)
print('\nUB\n', df_ub)
print('\nTime\n', df_time)

if SAVE:
    clusters_variables = {'violations': store_violations, 'lb_th': store_lb_th, 'lb_aux': store_lb_aux , 'ub': store_ub,
            'solver_time_tot': store_tot_time, 'time_constr_modification': store_modification_time}
    variables_to_save = {'clusters_variables': clusters_variables, 'K_list': K_list, 'N_exp_for_confidence': N_exp_for_confidence}
    file_name = 'performance' + '_delta' + str(delta) + '_' + str(idx_experiment) + '.pkl'
    file_name = os.path.join('variables_ol', file_name)
    with open(file_name, 'wb') as file:
        pickle.dump(variables_to_save, file)
