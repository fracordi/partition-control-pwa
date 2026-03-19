import numpy as np
import pandas as pd
import warnings
from typing import Any, List
from scipy.special import comb


def solve_pp(X, Y, D1, D2, d, E1, E2, E3, e, e_feas, epsilon, delta, norm_type, pp_optimizer, constraint_modification,
             generator, samples, regions_params, boxes_params, N_samples_pp, partition_strategy):

    p_hat, nominal_scen_params, regions_new_params, boxes_new_params, clusters_params = generator.compute_partition_elements(
        samples, regions_params, boxes_params, N_samples_pp, traslate=False)
    check_probability_sum(p_hat)

    nominal_scen = generator.from_params_to_sin(nominal_scen_params)
    regions_new, boxes_new = generator.partition_from_params(boxes_new_params, nominal_scen, traslate=True)

    if partition_strategy == 'clustering':
        pp_optimizer.set_tightening(D2, regions=regions_new)
        pp_optimizer.set_gamma(D2, regions=regions_new)
    else:
        pp_optimizer.set_tightening(D2, boxes=boxes_new)
        pp_optimizer.set_gamma(D2, boxes=boxes_new)


    cost, x, y, mu, solver_time_pp, compile_time_pp, status = pp_optimizer.optimize(X, Y, D1, D2, d, E1, E2, E3, e,
                                                                                    e_feas, nominal_scen, p_hat,
                                                                                    epsilon,
                                                                                    delta, norm_type=norm_type,
                                                                                    constraint_modification=constraint_modification)
    return cost, x, y, mu, solver_time_pp, compile_time_pp, status


def solve_robust(X, Y, D1, D2, d, E1, E2, E3, e, e_feas, norm_type, rob_optimizer,
             generator, samples_params, regions_params, boxes_params, N_samples_pp,
                 cost_type='avg'):
    if cost_type != 'avg' and cost_type != 'nominal':
        raise ValueError('cost_type must be either "avg" or "nominal"')

    p_hat, nominal_scen_params, regions_new_params, boxes_new_params, clusters_params = generator.compute_partition_elements(
        samples_params, regions_params, boxes_params, N_samples_pp, traslate=False)
    check_probability_sum(p_hat)

    samples = generator.from_params_to_sin(samples_params)
    nominal_scen = generator.from_params_to_sin(nominal_scen_params)
    regions_new, boxes_new = generator.partition_from_params(boxes_new_params, nominal_scen, traslate=True)

    rob_optimizer.set_tightening(D2, boxes=boxes_new)

    cost, x, y, solver_time_pp, compile_time_pp, status = rob_optimizer.optimize_robust(X, Y, D1, D2, d, E1, E2, E3, e,
                                                                                     e_feas, nominal_scen, samples,
                                                                                     norm_type=norm_type, cost_type=cost_type)
    return cost, x, y, solver_time_pp, compile_time_pp, status

def compute_stats(arr: np.ndarray, experiments: List[Any] = None, flatten=False, axis=1):
    """
    Compute min, max, average, and standard deviation along axis=1 of the array.
    Returns a DataFrame with statistics as rows and experiments as columns.
    """

    if flatten:
        arr = arr.flatten()[arr.flatten() >= 0]
        stats = {
            'min': round(float(np.min(arr)), 4),
            'max': round(float(np.max(arr)), 4),
            'avg': round(float(np.mean(arr)), 4),
            'std': round(float(np.std(arr)), 4)
        }
    else:
        arr = np.where(arr >= 0, arr, np.nan)
        stats = {
            'min': np.nanmin(arr, axis=axis),
            'max': np.nanmax(arr, axis=axis),
            'avg': np.nanmean(arr, axis=axis),
            'std': np.nanstd(arr, axis=axis)
        }
    # Create a DataFrame where each row is a statistic and each column is an experiment
    if experiments is not None:
        return pd.DataFrame(stats, index=experiments).T
    else:
        return stats


def check_probability_sum(p_hat: np.ndarray, tol: float = 1e-6) -> None:
    """
    Checks if probabilities sum to one. Issues a warning if they do not.
    """
    total = np.sum(p_hat)
    if abs(total - 1) > tol:
        warnings.warn(f"Probabilities do not sum to 1, but to {total}", UserWarning)


def filter_feasible_experiments(arr: np.ndarray) -> np.ndarray:
    """
    Filters out experiments where violations are -1 (infeasible experiments).
    Returns a list of filtered arrays.
    """
    mask = arr >= 0
    return arr[:, mask]


def solve_binomial_sum(n_x: int, epsilon: float, beta: float, num_modes: int, max_iter: int = 1000) -> int:
    """
    Solves the equation:
    Z sum_{i=0}^{d-1} binom(N, i) * epsilon^i * (1-epsilon)^(N-i) = beta
    for N using the bisection method.

    Args:
    - d (int): Number of decision variables of related optimization problem.
    - epsilon (float): Risk parameter.
    - beta (float): Confidence parameter.
    - tol (float): The tolerance for convergence.
    - max_iter (int): Maximum number of iterations.

    Returns:
    - N (float): The solution for N
    """
    def binomial_sum(N, num_modes):
        """Computes the summation for a given N."""
        return sum(comb(N, i) * (epsilon ** i) * ((1 - epsilon) ** (N - i)) for i in range(n_x)) * num_modes
    # Initial bounds for bisection
    high = 2 / epsilon * (n_x - 1 + np.log(num_modes / beta))
    N_ini = round(high / 4)
    # Bisection method
    for _ in range(max_iter):
        beta_eval = binomial_sum(N_ini, num_modes)
        if beta_eval <= beta:
            return N_ini
        N_ini += 1
    raise ValueError('Insufficient number of iterations.')


def get_random_matrices(seed, n_instances, num_modes, n_constr, n_x, n_uncertainty, uncertainty_in_cost, ensure_feasibility):
    rng = np.random.default_rng(seed)

    D1_all, D2_all, d_all = [], [], []
    E1_all, E2_all, E3_all, e_all = [], [], [], []

    for _ in range(n_instances):

        D1 = [rng.uniform(-1,1,(n_constr,n_x)) for _ in range(num_modes)]
        D2 = [rng.uniform(-1,1,(n_constr,n_uncertainty)) for _ in range(num_modes)]
        d  = [rng.uniform(-1,1,(n_constr,)) for _ in range(num_modes)]

        E13 = rng.uniform(-1,1,(n_x,2*n_x))
        E1 = E13[:,:n_x]
        E3 = E13[:,n_x:]

        if uncertainty_in_cost:
            E2 = rng.uniform(-1,1,(n_x,n_uncertainty))
        else:
            E2 = np.zeros((n_x,n_uncertainty))

        e = rng.uniform(-1,1,(n_x,))

        D1_all.append(D1)
        D2_all.append(D2)
        d_all.append(d)

        E1_all.append(E1)
        E2_all.append(E2)
        E3_all.append(E3)
        e_all.append(e)

    e_feas = 1 if ensure_feasibility else 0

    return D1_all, D2_all, d_all, E1_all, E2_all, E3_all, e_all, e_feas



