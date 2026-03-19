import numpy as np
from controller import PartitionControllerPwa
from uncertainty_utils import TruncatedGaussianNoise
from config import TRCGAUSS_PARAMS, SYS_PARAMS
import polytope as pt
import warnings
import pickle
import time
import os


#%% Simulations settings
SAVE = False
SOLVER = 'GUROBI'
PARTITIONING = 'adpt'
UNCERTAINTY = 'gauss'
seed = 1
idx_experiment = '_1'
N_exp = 500
M_grid = 3
n_to_split = 1
proportions = np.ones((n_to_split, M_grid))
K_to_split = M_grid ** n_to_split
slack = 1e-6
K_ini = 1
print_frequency = 20    # print every print_frequency experiments

print(f'\n### Partitioning: {PARTITIONING}, K: {K_to_split}, N_exp: {N_exp}, Save: {SAVE} ###')

# Load sys params
(n_x, n_u, n_uncertainty, eps, delta, beta, norm_type, N_horizon, N_control, T_cl, eps_nom, A, B, C, v, E, e, modes,
 Hx, hx, Hu, hu, X, U, Q, R, x_ref, u_ref, x0_ini, stab_threshold) = SYS_PARAMS.values()

#%% Controller and uncertainty
pp_controller = PartitionControllerPwa(n_x, n_u, n_uncertainty, N_horizon, A, B, C, E, e, modes, X, U, Q, R, x_ref, u_ref,
                                          norm_type, eps_nom, SOLVER='GUROBI')
pp_controller.set_compact_matrices()
generator = TruncatedGaussianNoise(TRCGAUSS_PARAMS)
lb_ini, ub_ini = generator.get_domain(N_horizon)
uncertainty_domain = np.array([lb_ini, ub_ini]).transpose()
uncertainty_domain_pt = pt.box2poly(uncertainty_domain)
boxes_ini = [uncertainty_domain]
regions_ini = [{'A': uncertainty_domain_pt.A, 'b': uncertainty_domain_pt.b.reshape(-1, 1)}]

#%% Closed-loop simulation
N_samples = int((K_to_split * np.log(2) + np.log(1 / beta)) / (2 * delta ** 2)) + 1
samples_cl_batches = generator.get_samples(1, T_cl, seed=seed, n_batches=N_exp)
samples_batches = generator.get_samples(N_samples, N_horizon, seed=seed, n_batches=N_exp)

# Store matrices
cost_matrix = np.ndarray((N_exp, T_cl))
u_matrix = np.ndarray((N_exp, T_cl))
u_list = []
time_solver_matrix = np.ndarray((N_exp, T_cl))
time_online_partitioning_matrix = np.ndarray((N_exp, T_cl))
time_tightening_matrix = np.ndarray((N_exp, T_cl))
store_sampled_trajectories = []

for i in range(N_exp):
    # Init. params for closed-loop simulation
    time_sim_start, x_t, K, solver_time_i = time.time(), x0_ini.copy(), K_ini, 0
    regions, boxes = regions_ini, boxes_ini
    eta_cl = samples_cl_batches[i].T
    samples = samples_batches[i]
    u_matrix = np.ndarray((T_cl, N_control))
    pp_controller.set_modes_indices_x0(x_t)
    x_traj = np.ndarray((T_cl+1, n_x))
    x_traj[0, :] = x0_ini.T[0]

    for t in range(1, T_cl+1):
        # Partition
        p_hat, nominal_scen, regions_new, boxes_new, clusters = generator.compute_partition_elements(samples,
                                                                                                     regions, boxes,
                                                                                                     N_samples,
                                                                                                     traslate=True)
        if np.sum(p_hat) > 1 + 1e-6 or np.sum(p_hat) < 1 - 1e-6:
            warnings.warn("Warning: probabilities do not sum to 1")
            print(np.sum(p_hat))
            print(p_hat)
        # Tightening time
        time_start_mod = time.time()
        pp_controller.set_tightening(boxes=boxes_new)
        time_tightening_matrix[i, t-1] = time.time() - time_start_mod

        # Solve MPC problem
        cost_pp, u_feas, mu_val, alpha_val, time_solver, problem_status = pp_controller.optimize(x_t, delta, p_hat,
                                                                                        nominal_scen, 'tight',
                                                                                                 N_control=N_control,
                                                                                                 slack=slack)
        # Store solution
        u_matrix[t-1, :] = u_feas.T[0]
        time_solver_matrix[i, t-1] = time_solver
        solver_time_i += time_solver

        # Next step ahead & update possible modes
        for h in range(len(modes)):
            if np.all(E[h] @ x_t <= e[h]):
                x_next = A[h] @ x_t + B @ u_feas[0:1] + C @ eta_cl[(t-1)*n_uncertainty:t*n_uncertainty].reshape(-1, 1)
                break
        x_t = x_next
        x_traj[t, :] = x_next.T[0]
        pp_controller.set_modes_indices_x0(x_t)
        # This is \|Q@x_{t+1}\| + \|R@u_t\|, since x_0 is just an offset
        cost_at_t = np.linalg.norm(Q @ x_t, norm_type) + np.linalg.norm(R @ u_feas[0:1], norm_type)
        cost_matrix[i, t-1] = cost_at_t

        ## From MPC result at time t, construct partition for next time step:
        # Construct u and \eta sequence by shifting and completing. Uncertainty: pick mid point for last entry
        u_candidate = np.vstack([u_feas, u_feas[-n_u]])[n_u:, :]
        scen_last = boxes_ini[0][-n_uncertainty:].mean(axis=1)
        scen_candidate = []
        for scen in nominal_scen:
            scen_cand = np.hstack([scen, scen_last])[n_uncertainty:]
            scen_candidate.append(scen_cand)
        scen_candidate = np.vstack(scen_candidate)

        # Find coordinates of uncertainty to split, for adpt. split: if any, and if x not yet at equilibrium, split
        idx_to_split = pp_controller.get_sensitive_uncertainty(x_next, u_candidate, scen_candidate, n_critical=3, n_influent=2)
        if np.size(idx_to_split) > 0 and np.linalg.norm(x_next, 2) > stab_threshold:
            idx_to_split = idx_to_split[0:n_to_split]
            regions, boxes = generator.split_gridding(lb_ini, ub_ini, M=M_grid, split_idx=idx_to_split, proportions=proportions)  ## Gridding
        else:
            regions, boxes = regions_ini, boxes_ini

    if (i + 1) % print_frequency == 0:
        print(f'\n### Experiment {i + 1} / {N_exp} ###')
        cost_matrix_avg = cost_matrix[:i+1, :].mean(axis=0)
        print('AVG. COST', cost_matrix_avg)
        print('SUM COST', np.sum(cost_matrix_avg))
        avg_cost_bound = np.cumsum(cost_matrix_avg) / np.arange(1, T_cl + 1)
        print('AVG. CUM COST BOUND =', avg_cost_bound)
        print('Solver time tot', solver_time_i)
        print('Elapsed time tot', time.time() - time_sim_start)
    store_sampled_trajectories.append(x_traj)
    u_list.append(u_matrix)
t_cl_range = np.arange(0, T_cl+1)


cost_matrix_avg = np.mean(cost_matrix, axis=0)
print('AVG. COST', cost_matrix_avg)
print('SUM COST', np.sum(cost_matrix_avg))
avg_cost_bound = np.cumsum(cost_matrix_avg) / np.arange(1, T_cl + 1)
print('AVG. CUM COST BOUND =', avg_cost_bound)


if SAVE:
    # Save variables to a file
    file_name = ('closed_loop_' + PARTITIONING + '_' + UNCERTAINTY + '_KtoSplit' + str(K_to_split)
                 + '_delta' + str(delta) + '_Nexp' + str(N_exp) + '_Tcl' + str(T_cl)
                 + idx_experiment + '.pkl')
    file_name = os.path.join('variables_cl', file_name)
    variables_to_save = {'cost_matrix': cost_matrix, 'solver_time_matrix': time_solver_matrix,
                         'partitioning_time_matrix': time_online_partitioning_matrix,
                         'time_tightening_matrix': time_tightening_matrix,
                         'K_to_split': K_to_split, 'Tcl': T_cl,
                         'N_exp': N_exp, 'sys_params': SYS_PARAMS, 'x_traj_list': store_sampled_trajectories,
                         'u_matrix': u_list, 'unc_params': TRCGAUSS_PARAMS, 'seed': seed}
    with open(file_name, 'wb') as file:
        pickle.dump(variables_to_save, file)
