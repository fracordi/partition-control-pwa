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
PARTITIONING = 'kmns'
UNCERTAINTY = 'gauss'
seed = 1
N_exp = 500
K = 3
n_to_split = 1
slack = 1e-6
print_frequency = 20    # print every print_frequency experiments

print(f'\n### Partitioning: {PARTITIONING}, K: {K}, N_exp: {N_exp}, Save: {SAVE} ###')

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
N_samples = int((K * np.log(2) + np.log(1 / beta)) / (2 * delta ** 2)) + 1
samples_cl_batches = generator.get_samples(1, T_cl, seed=seed, n_batches=N_exp)
sample_clustering_batches = generator.get_samples(1000, N_horizon, seed=seed, n_batches=N_exp)
samples_batches = generator.get_samples(N_samples, N_horizon, seed=seed, n_batches=N_exp)

# Store matrices
u_list = []
cost_matrix = np.ndarray((N_exp, T_cl))
time_solver_matrix = np.ndarray((N_exp, T_cl))
time_partitioning_matrix = np.ndarray(N_exp)
time_tightening_matrix = np.ndarray(N_exp)
store_sampled_trajectories = []

for i in range(N_exp):
    # Init. params for closed-loop simulation
    time_sim_start, x_t, K, solver_time_i = time.time(), x0_ini.copy(), K, 0
    eta_cl = samples_cl_batches[i].T
    pp_controller.set_modes_indices_x0(x_t)
    x_traj = np.ndarray((T_cl+1, n_x))
    x_traj[0, :] = x0_ini.T[0]

    # These are the samples for probability estimation
    sample_clustering = sample_clustering_batches[i]
    regions = generator.my_kmeans_partition(sample_clustering, K, uncertainty_domain_pt, N_horizon * n_uncertainty,
                                            random_state=0)
    samples = samples_batches[i]
    time_partitioning_star = time.time()
    p_hat_K, nominal_scen_K, regions_new_K, _, _ = generator.compute_partition_elements(samples, regions,
                                                                                                        None,
                                                                                                        N_samples,
                                                                                                        traslate=True)
    # to be used when stab_threshold is reached
    p_hat_ini, nominal_scen_ini, _, boxes_new_ini, _ = generator.compute_partition_elements(
        samples, regions_ini,
        boxes_ini,
        N_samples,
        traslate=True)

    # Tightening and partitioning time
    time_partitioning_matrix[i] = time.time() - time_partitioning_star
    time_tightening_star = time.time()
    tightening_K = pp_controller.set_tightening(regions=regions_new_K)
    time_tightening_matrix[i] = time.time() - time_tightening_star
    u_matrix = np.ndarray((T_cl, N_control))
    p_hat, nominal_scen = p_hat_K, nominal_scen_K
    threshold_reached = False

    for t in range(1, T_cl+1):
        if np.sum(p_hat) > 1 + 1e-6 or np.sum(p_hat) < 1 - 1e-6:
            warnings.warn("Warning: probabilities do not sum to 1")
            print(np.sum(p_hat))
            print(p_hat)

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
        # print('x_next', x_next.T[0] )
        pp_controller.set_modes_indices_x0(x_t)
        # This is \|Q@x_{t+1}\| + \|R@u_t\|, since x_0 is just an offset
        cost_at_t = np.linalg.norm(Q @ x_t, norm_type) + np.linalg.norm(R @ u_feas[0:1], norm_type)
        cost_matrix[i, t-1] = cost_at_t

        # if threshold is reached, reduce K. If is exceeded after it is reached, go back to original partition.
        if np.linalg.norm(x_t, 2) <= stab_threshold and not threshold_reached:
            pp_controller.set_tightening(boxes=boxes_new_ini)
            p_hat, nominal_scen = p_hat_ini, nominal_scen_ini
            threshold_reached = True
        elif np.linalg.norm(x_t, 2) > stab_threshold and threshold_reached:
            pp_controller.set_tightening(tightening=tightening_K)
            p_hat, nominal_scen = p_hat_K, nominal_scen_K
            threshold_reached = False

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

#%%
if SAVE:
    # Save variables to a file
    file_name = ('closed_loop_' + PARTITIONING + '_' + UNCERTAINTY + '_KtoSplit' + str(K)
                 + '_delta' + str(delta) + '_Nexp' + str(N_exp)  + '_Tcl' + str(T_cl) + '.pkl')
    file_name = os.path.join('variables_cl', file_name)
    variables_to_save = {'cost_matrix': cost_matrix, 'solver_time_matrix': time_solver_matrix,
                         'k_means_time_matrix': time_partitioning_matrix,
                         'time_tightening_matrix': time_tightening_matrix,
                         'K_to_split': K, 'Tcl': T_cl, 'x_traj_list': store_sampled_trajectories,
                         'u_matrix': u_list, 'N_exp': N_exp, 'sys_params': SYS_PARAMS,
                         'unc_params': TRCGAUSS_PARAMS, 'seed': seed}
    with open(file_name, 'wb') as file:
        pickle.dump(variables_to_save, file)
