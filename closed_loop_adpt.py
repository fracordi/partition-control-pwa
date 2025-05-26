import numpy as np
from controller import PartitionControllerPwa
from uncertainty import SinusoidalDisturbance
from config import SYS_PARAMS, UNC_PARAMS
import polytope as pt
import warnings
import pickle
import time
import os

from closed_loop_kmns import idx_experiment

#%% Simulations settings

SAVE = False
SOLVER = 'GUROBI'
idx_experiment = 1
K_to_split = 4
K_ini = 1
T_cl = 80
N_exp = 125
N_samples_test = 100000
delta = 0.05
beta = 1e-6
print_frequency = 10    # print every print_frequency experiments
REDUCE_K = True
stab_threshold = 0.3
UNC_PARAMS = UNC_PARAMS['sinusoidal']

#%% Define variables for closed loop simulations
(n_x, n_u, n_uncertainty, norm_type, N_horizon, eps_nom, modes, Hx, hx, Hu, hu, Q, R, x_ref, u_ref, x0_ini) \
    = SYS_PARAMS.values()
X, U = {'A': Hx, 'b': hx}, {'A': Hu, 'b': hu}

pp_controller = PartitionControllerPwa(n_x, n_u, n_uncertainty, N_horizon, modes, X, U, Q, R, x_ref, u_ref,
                                          norm_type, eps_nom, SOLVER=SOLVER)

generator = SinusoidalDisturbance(UNC_PARAMS)
lb_ini, ub_ini = generator.get_domain(N_horizon)
uncertainty_domain = np.array([lb_ini, ub_ini]).transpose()
uncertainty_domain_pt = pt.box2poly(uncertainty_domain)
cost_matrix = np.ndarray((N_exp, T_cl))
time_solver_matrix = np.ndarray((N_exp, T_cl))
time_initial_partitioning_matrix = np.ones((N_exp,)) * -1
time_online_partitioning_matrix = np.ndarray((N_exp, T_cl))
time_tightening_matrix = np.ndarray((N_exp, T_cl))
cost_sum = 0

boxes = [uncertainty_domain]
boxes_ini = boxes
regions_ini = [{'A': uncertainty_domain_pt.A, 'b': uncertainty_domain_pt.b.reshape(-1, 1)}]
regions = regions_ini

#%% Closed-loop simulation
for i in range(N_exp):
    time_start, x_t, cost_cl, cost_at_t, K, solver_time_i = time.time(), x0_ini.copy(), 0, 1000, K_ini, 0
    for t in range(T_cl):
        pp_controller.set_compact_matrices(x_t)

        N_samples = int((K * np.log(2) + np.log(1 / beta)) / (2 * delta ** 2)) + 1
        # These are the samples for probability estimation
        samples = generator.get_samples(N_samples, N_horizon, t)

        p_hat, nominal_scen, polytopes, regions_new, boxes_new, clusters = generator.compute_partition_elements(samples,
                                                                                                     regions, boxes,
                                                                                                     N_samples)
        if np.sum(p_hat) > 1 + 1e-6 or np.sum(p_hat) < 1 - 1e-6:
            warnings.warn("Warning: probabilities do not sum to 1")
            print(np.sum(p_hat))
            print(p_hat)
        time_start_mod = time.time()
        pp_controller.set_tightening(nominal_scen, boxes=boxes_new)
        time_tightening_matrix[i, t] = time.time() - time_start_mod
        cost_pp, u_feas, mu_val, alpha_val, time_solver, problem_status = pp_controller.optimize(x_t, delta, p_hat,
                                                                                        nominal_scen, 'tight')
        time_solver_matrix[i, t] = time_solver
        solver_time_i += time_solver

        #adpt partitioning
        if K_ini > 1:
            raise ValueError('Adaptive splitting only works with K_ini = 1.')
        # Pick worst time step and split into half.
        time_to_split = pp_controller.get_worst_case(x_t, u_feas, samples)
        if time_to_split >= 1 and (cost_at_t > stab_threshold or REDUCE_K is False):
            K = K_to_split
            idxs_uncertainty_at_tsplit = np.arange(n_uncertainty * time_to_split, n_uncertainty * (time_to_split + 1))
            time_partitioning = time.time()
            regions, boxes = generator.split_iteratively(lb_ini, ub_ini, K, idxs_uncertainty_at_tsplit)
            time_online_partitioning_matrix[i, t] = time.time() - time_partitioning
            K = len(regions)
        else:
            # If worst case is 1st predicted state: fall back to robust solution
            K, regions, boxes = 1, regions_ini, boxes_ini
        real_disturbance = generator.get_samples(1, 1, t).transpose()
        for h in range(len(modes)):
            if np.all(modes[h]['E'] @ x_t <= modes[h]['e']):
                x_next = (modes[h]['A'] @ x_t + modes[h]['B'] @ u_feas[0:1] +
                          modes[h]['C'] @ real_disturbance[0:n_uncertainty] + modes[h]['v'])
                break
        x_t = x_next
        cost_at_t = np.linalg.norm(Q @ x_t, norm_type) + np.linalg.norm(R @ u_feas[0:1], norm_type)
        cost_matrix[i, t] = cost_at_t
    if (i + 1) % print_frequency == 0:
        print(f'\nExperiment {i + 1} / {N_exp}')
        print('cum_cost_avg / Tcl, partial', cost_matrix[:i+1, :].mean(axis=1).mean())
        print('Solver time tot', solver_time_i)
        print('Elapsed time tot', time.time() - time_start)

cost_avg = cost_sum / N_exp
print('cum_cost_avg / Tcl', cost_avg / T_cl)

cost_matrix_avg = np.mean(cost_matrix, axis=0)
avg_cost_bound = np.cumsum(cost_matrix_avg) / np.arange(1, T_cl + 1)
print('AVG. COST BOUND =', avg_cost_bound)


if SAVE:
    # Save variables to a file
    file_name = 'closed_loop_adpt_K' + str(K_to_split) + '_' + str(idx_experiment) + '.pkl'
    file_name = os.path.join('variables_cl', file_name)
    variables_to_save = {'cost_matrix': cost_matrix, 'solver_time_matrix': time_solver_matrix,
                         'partitioning_time_matrix': time_online_partitioning_matrix,
                         'time_tightening_matrix': time_tightening_matrix}
    with open(file_name, 'wb') as file:
        pickle.dump(variables_to_save, file)
