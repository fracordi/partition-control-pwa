import numpy as np
from config import SYS_PARAMS, SINUSOIDAL_PARAMS
from utils import filter_feasible_experiments, compute_stats
from controller import RandomizedController
from uncertainty import SinusoidalDisturbance
import pickle
import os


#%% Params for simulations

SAVE = False
idx_experiment = 1
SOLVER = 'GUROBI'
N_samples_test = 100000
print_frequency = 10    # print every print_frequency experiments

N_exp_for_confidence = 33
N_samples_list = [5, 10, 20, 50, 100, 200, 500, 1000]
beta = 1e-4


store_cost = np.empty((len(N_samples_list), N_exp_for_confidence))
store_tot_time = np.empty((len(N_samples_list), N_exp_for_confidence))
store_violations = np.empty((len(N_samples_list), N_exp_for_confidence))

generator = SinusoidalDisturbance(SINUSOIDAL_PARAMS)
(n_x, n_u, n_uncertainty, norm_type, N_horizon, eps_nom, modes, Hx, hx, Hu, hu, Q, R, x_ref, u_ref, x0) = SYS_PARAMS.values()
X, U = {'A': Hx, 'b': hx}, {'A': Hu, 'b': hu}

ra_controller = RandomizedController(n_x, n_u, n_uncertainty, N_horizon, modes,
                                     X, U, Q, R, x_ref, u_ref, norm_type, eps_nom, SOLVER)
ra_controller.set_compact_matrices(x0)

samples_test_violation = generator.get_samples(N_samples_test, N_horizon)

for i, N_samples in enumerate(N_samples_list):
    print(f'\n\n ### Nsamples = {N_samples} ###')
    for h in range(N_exp_for_confidence):
        samples = generator.get_samples(N_samples, N_horizon)
        cost, u_feas, _, time_tot, _ = ra_controller.optimize(x0, samples)
        if cost is not None:
            violation = ra_controller.get_empirical_violation(x0, u_feas, samples_test_violation)
        else:
            violation = -1
        store_violations[i, h] = violation
        store_cost[i, h] = cost
        store_tot_time[i, h] = time_tot

        if (h + 1) % print_frequency == 0:
            print(f'\nExperiment {h + 1} / {N_exp_for_confidence}')
            print(f'Avg. Violation: {store_violations[i, :h + 1].mean()}')
            print(f'Avg. Cost: {store_cost[i, :h + 1].mean()}')
            print(f'Avg. Time: {store_tot_time[i, :h + 1].mean()}')

store_violations, store_cost, store_tot_time = filter_feasible_experiments(
    store_violations, [store_violations, store_cost, store_tot_time]
)
df_viol = compute_stats(store_violations, N_samples_list)
df_cost = compute_stats(store_cost, N_samples_list)
df_time = compute_stats(store_tot_time, N_samples_list)

print('\n\n ### Results ###')
print('\nViolation\n', df_viol)
print('\nCost\n', df_cost)
print('\nTime\n', df_time)

if SAVE:
    confidence_RA_variables = {'violations': store_violations, 'cost': store_cost, 'solver_time_tot': store_tot_time}
    variables_to_save = {'confidence_RA_variables': confidence_RA_variables, 'N_exp_for_confidence': N_exp_for_confidence}
    file_name = 'confidence_RA' + str(idx_experiment) + '.pkl'
    file_name = os.path.join('variables_ol', file_name)
    with open(file_name, 'wb') as file:
        pickle.dump(variables_to_save, file)

