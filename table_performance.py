import pickle
import math
import numpy as np

for delta_str in ['0.05', '0.01']:
    name_file_to_open = './variables_numerical/numeric_performance_partition_nx10_nunc10_N_exp_confidence500_delta' + delta_str + '.pkl'
    with open(name_file_to_open, 'rb') as file:
        clusters_matrices = pickle.load(file)
    M_grid_list = clusters_matrices['M_grid_list']
    beta = clusters_matrices['beta']
    delta = clusters_matrices['delta']
    print(f'\n\n\n### ### Delta = {delta} ### ###')
    for k, M_grid in enumerate(M_grid_list):
        K = math.prod(M_grid)
        cost_lb_aux = clusters_matrices['store_cost_lb_aux'].mean(axis=1)
        cost_ub = clusters_matrices['store_cost_ub'].mean(axis=1)
        empirical_cost = clusters_matrices['store_empirical_cost'].mean(axis=1)
        empirical_violation = clusters_matrices['store_empirical_violation'].mean(axis=1)
        solver_time = clusters_matrices['store_solver_time_pp'].mean(axis=1) + clusters_matrices['store_solver_time_rp'].mean(axis=1)
        N = int((K * np.log(2) + np.log(1 / beta)) / (2 * delta ** 2)) + 1
        print(f'\n### K = {K} ###')
        print(f'# LB aux: {cost_lb_aux[k]}')
        print(f'# UB: {cost_ub[k]}')
        print(f'# Empirical cost: {empirical_cost[k]}')
        print(f'# Empirical violation: {empirical_violation[k]}')
        print(f'# Solver time: {solver_time[k]}')
        print(f'# N: {N}')


### MC ###
name_file_to_open = './variables_numerical/numeric_mc_nx10_nunc10_N_exp_confidence500_delta0.0.pkl'
with open(name_file_to_open, 'rb') as file:
    mc_matrices = pickle.load(file)
empirical_cost_mc = mc_matrices['store_empirical_cost'].mean(axis=0)
empirical_violation_mc = mc_matrices['store_empirical_violation'].mean(axis=0)
solver_time_mc = mc_matrices['store_solver_time'].mean(axis=0)

print('\n\n\n### ### Sample average approximation ### ###')
print(f'# Empirical cost: {empirical_cost_mc}')
print(f'# Empirical violation: {empirical_violation_mc}')
print(f'# Solver time: {solver_time_mc}')



### Robust (Margellos et al.) ###
name_file_to_open = './variables_numerical/numeric_robust_nx10_nunc10_eps0.15_Nsamples287_N_exp_confidence500_cost_avg.pkl'
with open(name_file_to_open, 'rb') as file:
    robust_matrices = pickle.load(file)
empirical_cost_robust = robust_matrices['store_empirical_cost'].mean(axis=0)
empirical_violation_robust = robust_matrices['store_empirical_violation'].mean(axis=0)
solver_time_robust = robust_matrices['store_solver_time'].mean(axis=0)

print('\n\n\n### ### Robust-random approach ### ###')
print(f'# Empirical cost: {empirical_cost_robust}')
print(f'# Empirical violation: {empirical_violation_robust}')
print(f'# Solver time: {solver_time_robust}')