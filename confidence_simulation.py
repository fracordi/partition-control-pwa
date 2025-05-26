import numpy as np
from config import SYS_PARAMS, SINUSOIDAL_PARAMS
from utils import filter_feasible_experiments, compute_stats, check_probability_sum
import polytope as pt
from controller import PartitionControllerPwa
from uncertainty import SinusoidalDisturbance
import time
import pickle
import os


#%% Params for simulations

SAVE = False
idx_experiment = 1
SOLVER = 'GUROBI'
N_samples_test = 100000
print_frequency = 10    # print every print_frequency experiments
N_exp_for_confidence = 250
N_samples_list = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
K_ini = 5
delta = 0.1


store_cost = np.empty((len(N_samples_list), N_exp_for_confidence))
store_violations = np.empty((len(N_samples_list), N_exp_for_confidence))
store_solver_time = np.empty((len(N_samples_list), N_exp_for_confidence))
store_tightening_time = np.empty((len(N_samples_list), N_exp_for_confidence))

generator = SinusoidalDisturbance(SINUSOIDAL_PARAMS)
(n_x, n_u, n_uncertainty, norm_type, N_horizon, eps_nom, modes, Hx, hx, Hu, hu, Q, R, x_ref, u_ref, x0) = SYS_PARAMS.values()
X, U = {'A': Hx, 'b': hx}, {'A': Hu, 'b': hu}

# Get uncertainty domain and create initial polytope box.
lb, ub = generator.get_domain(N_horizon)
uncertainty_box = np.array([lb, ub]).transpose()
uncertainty_box_pt = pt.box2poly(uncertainty_box)

# Compute initial partition regions.
regions, boxes = generator.box_partitioning(K_ini, lb, ub)

# Set up the controller.
pp_controller = PartitionControllerPwa(n_x, n_u, n_uncertainty, N_horizon, modes,
                                       X, U, Q, R, x_ref, u_ref, norm_type, eps_nom, SOLVER)
pp_controller.set_compact_matrices(x0)

samples_test_violation = generator.get_samples(N_samples_test, N_horizon)


for i, N_samples in enumerate(N_samples_list):
    print(f'\n\n ### Nsamples = {N_samples} ###')
    for h in range(N_exp_for_confidence):
        samples = generator.get_samples(N_samples, N_horizon)
        p_hat, nominal_scen, polytopes, regions_new, boxes_new, clusters = generator.compute_partition_elements(
            samples, regions, boxes, N_samples)
        check_probability_sum(p_hat)
        time_tight_start = time.time()
        pp_controller.set_tightening(nominal_scen, boxes=boxes_new)
        store_tightening_time[i, h] = time.time() - time_tight_start

        cost_pp, u_feas, _, _, time_ub, _ = pp_controller.optimize(x0, delta, p_hat, nominal_scen, 'tight')
        if cost_pp is not None:
            violation = pp_controller.get_empirical_violation(x0, u_feas, samples_test_violation)
        else:
            violation = -1
        store_violations[i, h] = violation
        store_cost[i, h] = cost_pp
        store_solver_time[i, h] = time_ub
        if (h + 1) % print_frequency == 0:
            print(f'\nExperiment {h + 1} / {N_exp_for_confidence}')
            print(f'Avg. Violation: {store_violations[i, :h + 1].mean()}')
            print(f'Avg. Cost: {store_cost[i, :h + 1].mean()}')
            print(f'Avg. Time: {store_solver_time[i, :h + 1].mean()}')

# Filter out infeasible experiments.
store_violations, store_cost, store_solver_time = filter_feasible_experiments(
    store_violations, [store_violations, store_cost, store_solver_time]
)

df_viol = compute_stats(store_violations, N_samples_list)
df_cost = compute_stats(store_cost, N_samples_list)
df_time = compute_stats(store_solver_time, N_samples_list)

print(f'\n\n ### Results (K = {K_ini}) ###')
print('\nViolation\n', df_viol)
print('\nCost\n', df_cost)
print('\nTime\n', df_time)


if SAVE:
    confidence_variables = {'violations': store_violations, 'cost': store_cost,
                            'solver_time_tot': store_solver_time, 'tightening_time': store_tightening_time}
    variables_to_save = {'confidence_variables': confidence_variables, 'K_ini': K_ini}
    # Save variables to a file
    file_name = 'confidence_K' + str(K_ini) + '_' + str(idx_experiment) + '.pkl'
    file_name = os.path.join('variables_ol', file_name)
    with open(file_name, 'wb') as file:
        pickle.dump(variables_to_save, file)
