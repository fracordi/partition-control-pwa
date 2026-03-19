
import numpy as np
import cvxpy as cp
from itertools import combinations, product


class BasicOptimizer(object):
    """Basic class for optimizer"""

    def __init__(self, n_x, n_theta, SOLVER):
        self.n_x = n_x
        self.n_theta = n_theta
        self.SOLVER = SOLVER

    def penalty_np(self, x, norm_type):
        """
        Compute penalty according to chosen norm for np array
        """
        if norm_type == 2:
            return np.dot(x, x)
        elif norm_type == 1:
            return np.linalg.norm(x, norm_type)
        else:
            raise NotImplementedError

    def penalty_cp(self, x, norm_type):
        """
        Compute penalty according to chosen norm for cp variable
        """
        if norm_type == 2:
            return cp.sum_squares(x)
        elif norm_type == 1:
            return cp.norm(x, norm_type)
        else:
            raise NotImplementedError

    def cost_func_empirical(self, x, y, samples, E1, E2, E3, e, e_feas, norm_type=1):
        """
        Compute empirical cost of a solution (x,y)
        """
        Nsamples = len(samples)
        cost_list = [1 / Nsamples * self.penalty_np(E1 @ x + E2 @ samples[j] + e, norm_type) for j in range(Nsamples)]
        cost = np.sum(cost_list) + self.penalty_np(E3 @ x, norm_type) + e_feas * np.abs(y)
        return cost

    def cost_func(self, x, y, scenarios, p_hat, E1, E2, E3, e, e_feas, num_scenarios, norm_type=1):
        """
        Compute cost function for optimization problem
        """
        cost_list = [p_hat[j] * self.penalty_cp(E1 @ x + E2 @ scenarios[j] + e, norm_type) for j in range(num_scenarios)]
        return cp.sum(cost_list) + self.penalty_cp(E3 @ x, norm_type) + e_feas * cp.abs(y)

    def compute_empirical_metrics(self, x, y, samples, D1, D2, d, E1, E2, E3, e, e_feas, norm_type=1):
        """
        Compute empirical cost and violation
        """
        cost = self.cost_func_empirical(x, y, samples, E1, E2, E3, e, norm_type)
        Z_list = []

        samples_constraints = ((D1 @ x + d).reshape(-1, 1) + D2 @ samples.T).T + e_feas * y
        violations = (samples_constraints >= 0).astype(int)
        Z = np.any(violations == 1, axis=1).astype(int)
        Z_list.append(Z)

        Z_cat = np.vstack(Z_list)
        Z_min = Z_cat.min(axis=0)
        return cost, np.mean(Z_min)


class PartitionOptimizerLP(BasicOptimizer):
    """Class for partition-based optimization"""

    def __init__(self, n_x, n_theta, SOLVER):
        super().__init__(n_x, n_theta, SOLVER)
        self.tau = None
        self.gamma = None

    def _compute_correction_box(self, D, translated_boxes=None, correction='tightening'):
        """
        Compute tightening or relaxation for box regions.
        """
        num_scenarios = len(translated_boxes)
        sigma = []
        for j in range(num_scenarios):
            if correction == 'tightening':
                lower_bound = translated_boxes[j][:, 0]
                upper_bound = translated_boxes[j][:, 1]
                sgn = -1.
            else:
                lower_bound = -translated_boxes[j][:, 1]
                upper_bound = -translated_boxes[j][:, 0]
                sgn = 1.
            sigma_j = sgn * (np.maximum(D, 0) @ upper_bound + np.minimum(D, 0) @ lower_bound)
            sigma.append(sigma_j)
        return np.array(sigma)

    def _compute_correction_general(self, D, translated_regions=None, correction='tightening'):
        """
        Compute tightening or relaxation for polytopic regions.
        """
        num_scenarios = len(translated_regions)

        sigma = []
        dim_theta = D.shape[1]
        if correction == 'tightening':
            sgn = 1.
        else:
            sgn = -1.
        for j in range(num_scenarios):
            region = translated_regions[j]
            dim_tau = D.shape[0]
            theta_list = [cp.Variable(dim_theta) for _ in range(dim_tau)]
            cost_list = [sgn * D[i] @ theta_list[i] for i in range(dim_tau)]
            cost = cp.sum(cost_list)
            constr = [region['A'] @ theta_list[i] <= region['b'].squeeze() for i in range(dim_tau)]
            problem = cp.Problem(cp.Maximize(cost), constr)
            problem.solve(verbose=False, solver=self.SOLVER)
            sigma_j = np.array([(-sgn) * c.value for c in cost_list])
            sigma.append(sigma_j)
        return np.array(sigma)

    def set_tightening(self, D, regions: list = None, boxes: list = None) -> None:
        """
        Set constraint tightening for box or polyhedral regions.
        """
        if boxes is not None:
            self.tau = self._compute_correction_box(D, boxes, 'tightening')
        elif regions is not None:
            self.tau = self._compute_correction_general(D, regions, 'tightening')
        else:
            raise ValueError('Either specify polytopes or boxes for uncertainty sets.')

    def set_gamma(self, D, regions: list = None, boxes: list = None) -> None:
        """
        Set constraint relaxation from box or polyhedral regions.
        """
        if boxes is not None:
            self.gamma = self._compute_correction_box(D, boxes, 'relaxation')
        elif regions is not None:
            self.gamma = self._compute_correction_general(D, regions, 'relaxation')
        else:
            raise ValueError('Either specify polytopes or boxes for uncertainty sets.')


    def constraints(self, x, y, X, Y, mu, scenarios, p_hat, D1, D2, d, e_feas, slack, eps_modified, K):
        """
        Constraint for PP problem for probabilistic linear LP
        """
        constr = [X['A'] @ x <= X['b']]
        if e_feas:
            constr += [Y['A'] * y <= Y['b']]
        n_c = D1.shape[0]
        D1_vec = np.tile(D1, (K, 1))
        D2_vec = np.kron(np.eye(K), D2)
        scen_vec = scenarios.flatten()
        d_vec = np.tile(d, (1, K)).flatten()
        tau_vec = np.hstack([slack[j] for j in range(K)])
        mu_vec = cp.hstack([np.ones(n_c) * mu[j] for j in range(K)])
        constr += [D1_vec @ x + D2_vec @ scen_vec + d_vec + e_feas * y <= tau_vec + 1000 * (1 - mu_vec)]
        constr += [mu @ p_hat >= 1 - eps_modified]
        return constr


    def optimize(self, X, Y, D1, D2, d, E1, E2, E3, e, e_feas, scenarios, p_hat, eps, delta, norm_type=1,
                 constraint_modification='tight'):
        """
        Solve probabilistic linear LP via PP
        """
        num_scenarios = len(scenarios)
        x = cp.Variable(self.n_x)
        y = cp.Variable()
        mu = cp.Variable(num_scenarios, boolean=True)      # mu_j = 1 iff robust constraint satisfied for scenario j
        if constraint_modification == 'tight':
            eps_modified = eps - delta
            slack = self.tau
        elif constraint_modification == 'relax':
            eps_modified = eps + delta
            slack = self.gamma
        else:
            eps_modified = eps
            slack = np.zeros(num_scenarios)
        ## Cost
        cost = self.cost_func(x, y, scenarios, p_hat, E1, E2, E3, e, e_feas, num_scenarios, norm_type)

        ## Constraints
        constr = self.constraints(x, y, X, Y, mu, scenarios, p_hat, D1, D2, d, e_feas, slack, eps_modified,
                                  num_scenarios)
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(verbose=False, solver=cp.GUROBI)
        if problem.status not in ["infeasible_or_unbounded", "infeasible", "unbonded"]:
            y_val = 0 if e_feas == 0 else y.value
            return cost.value, x.value, y_val, mu.value, problem.solver_stats.solve_time, problem.compilation_time, problem.status
        else:
            return None, None, None, None, None, None, None


    def compute_lip_x(self, E1, E3, e_feas, n_x, norm_type):
        """
        Compute Lipschitz constant wrt x for performance bounds
        """
        if norm_type == 1:
            lip = np.sqrt(n_x) * (np.linalg.norm(E1, 1) + np.linalg.norm(E3, 1) + np.abs(e_feas))
        else:
            raise ValueError('norm_type must be 1 or (2 not yet implemented).')
        return lip

    def compute_lip_theta(self, E2, norm_type):
        """
        Compute Lipschitz constant wrt uncertainty for performance bounds
        """
        if norm_type == 1:
            lip = np.linalg.norm(E2, 1)
        else:
            raise ValueError('norm_type must be 1 or (2 not yet implemented).')
        return lip

    def compute_c(self, lip_theta, lip_x,  diam_unc, N_samples, beta, n_x, diam_x, r,
                  nominal_scen: list, clusters: list) -> float:
        """
        Compute error term due to partitioning.
        """

        c1 = np.sqrt( (lip_theta * diam_unc)**2 / (2*N_samples) * (np.log(1 / beta) + n_x * np.log(3*diam_x / r)) )
        c2 = 2 * lip_x * r
        c3 = 0
        for j in range(len(clusters)):
            Theta_hat = np.tile(nominal_scen[j], (len(clusters[j]), 1))
            c3 += np.sum(np.abs((Theta_hat - clusters[j])))
        c3 = c3 / N_samples
        c3 = c3 * lip_theta

        return c1, c2, c3

    def get_theoretical_hausdorff_distance(self, D1, num_scen: int) -> float:
        """
        Compute the minimum smallest singular value across all invertible square submatrices,
        of matrices in list D1
        """
        h_dist_list = []

        D = D1
        m, n = D.shape
        if m < n:
            raise ValueError("Matrix must have at least as many rows as columns to form square submatrices.")
        # Generate all combinations of rows to form square submatrices
        row_indices = range(m)
        sv_list = []
        for rows in combinations(row_indices, n):
            # Extract the square submatrix
            submatrix = D[np.array(rows), :]
            # Check if it's invertible (determinant non-zero)
            if np.abs(np.linalg.det(submatrix)) >= 1e-9:
                min_sv = np.min(np.linalg.svd(submatrix, compute_uv=False))
                # min_sv = 1 / np.linalg.norm(np.linalg.inv(submatrix), 2)
                sv_list.append(min_sv)
        sigma = min(sv_list)
        for j in range(num_scen):
            slack_norm_dist = np.linalg.norm(self.gamma[j] - self.tau[j], 2)
            h_dis_j = slack_norm_dist / sigma
            h_dist_list.append(h_dis_j)

        return max(h_dist_list)

    def get_performances(self, X, Y, D1, D2, d, E1, E2, E3, e, e_feas, nominal_scen, clusters, p_hat, epsilon, delta,
                         norm_type, N_samples, diam_unc, diam_x, r, beta, validation_set):
        """
        Get performance: upper and lower bound, empirical cost and probability of constraint violation,
        associated to PP problem
        """
        # PP
        cost_pp, x_pp, y_pp, mu_pp, solver_time_pp, compile_time_pp, status_pp = self.optimize(X, Y, D1, D2, d, E1, E2, E3, e, e_feas,
                                                                         nominal_scen, p_hat, epsilon,
                                                                         delta, norm_type=norm_type,
                                                                         constraint_modification='tight')
        # RP
        cost_rp, x_rp, y_rp, mu_rp, solver_time_rp, compile_time_rp, status_rp = self.optimize(X, Y, D1, D2, d, E1, E2, E3, e,
                                                                                           e_feas,
                                                                                           nominal_scen, p_hat, epsilon,
                                                                                           delta, norm_type=norm_type,
                                                                                           constraint_modification='relax')
        if x_pp is None:
            return -1, None, None, None, None, None, None, None, None
        else:
            num_scen = len(nominal_scen)
            lip_theta = self.compute_lip_theta(E2, norm_type)
            lip_x = self.compute_lip_x(E1, E3, e_feas, self.n_x, norm_type)
            hausd_dist = self.get_theoretical_hausdorff_distance(D1, num_scen)

            c1, c2, c3 = self.compute_c(lip_theta, lip_x, diam_unc, N_samples, beta, self.n_x, diam_x, r,
                  nominal_scen, clusters)
            c = c1 + c2 + c3
            cost_ub = cost_pp + c
            cost_lb_aux = cost_rp - c
            empirical_cost, empirical_violation = self.compute_empirical_metrics(x_pp, y_pp, validation_set, D1,
                                                                                              D2, d, E1, E2, E3, e,
                                                                                              e_feas,
                                                                                              norm_type=norm_type)

            bound_lhs = lip_x * hausd_dist + c
            cost_lb_th = cost_pp - bound_lhs

            return (cost_pp, cost_rp, cost_lb_th, cost_lb_aux, cost_ub, empirical_cost, empirical_violation,
                    solver_time_pp, compile_time_pp, solver_time_rp, compile_time_rp, c1, c2, c3)




class RandomizedOptimizer(BasicOptimizer):
    """Class for randomized linear programs"""
    def __init__(self, n_x, n_theta, SOLVER):
        super().__init__(n_x, n_theta, SOLVER)

    def constraints(self, x, y, X, Y, scenarios, D1, D2, d, e_feas):
        """
        Constraint given scenarios
        """
        K = len(scenarios)
        D1_vec = np.tile(D1, (K, 1))
        D2_vec = np.kron(np.eye(K), D2)
        scen_vec = scenarios.flatten()
        d_vec = np.tile(d, (1, K)).flatten()
        constr = [X['A'] @ x <= X['b']]
        if e_feas:
            constr += [Y['A'] * y <= Y['b']]
        constr += [D1_vec @ x + D2_vec @ scen_vec + d_vec + e_feas * y <= 0]
        return constr

    def optimize(self, X, Y, D1, D2, d, E1, E2, E3, e, e_feas, scenarios, norm_type=1):
        """
        Solve random program
        """
        num_scenarios = len(scenarios)
        x = cp.Variable(self.n_x)
        y = cp.Variable()

        ## Cost
        p_hat = [1/num_scenarios for _ in range(num_scenarios)]
        cost = self.cost_func(x, y, scenarios, p_hat, E1, E2, E3, e, e_feas, num_scenarios, norm_type)

        ## Constraints
        constr = self.constraints(x, y, X, Y, scenarios, D1, D2, d, e_feas)
        problem = cp.Problem(cp.Minimize(cost), constr)

        problem.solve(verbose=False, solver=cp.GUROBI)
        if problem.status not in ["infeasible_or_unbounded", "infeasible", "unbonded"]:
            y_val = 0 if e_feas == 0 else y.value
            return cost.value, x.value, y_val, problem.solver_stats.solve_time, problem.compilation_time, problem.status
        else:
            return None, None, None, None, None, None, None


class MCOptimizerLP(BasicOptimizer):
    """Class for Monte Carlo linear programs"""
    def __init__(self, n_x, n_theta, SOLVER):
        super().__init__(n_x, n_theta, SOLVER)
        self.tau = None

    def constraints(self, x, y, X, Y, mu, scenarios, p_hat, D1, D2, d, e_feas, eps, num_scenarios):
        """
        Constraint function formulated with N binaries
        """
        K = num_scenarios
        constr = [X['A'] @ x <= X['b']]
        if e_feas:
            constr += [Y['A'] * y <= Y['b']]
        n_c = D1.shape[0]
        D1_vec = np.tile(D1, (K, 1))
        D2_vec = np.kron(np.eye(K), D2)
        scen_vec = scenarios.flatten()
        d_vec = np.tile(d, (1, K)).flatten()
        mu_vec = cp.hstack([np.ones(n_c) * mu[j] for j in range(K)])

        constr += [D1_vec @ x + D2_vec @ scen_vec + d_vec + e_feas * y <= 1000 * (1 - mu_vec)]
        constr += [mu @ p_hat >= 1 - eps]
        return constr


    def optimize(self, X, Y, D1, D2, d, E1, E2, E3, e, e_feas, scenarios, p_hat, eps, norm_type=1):
        """
        Solve MC problem
        """
        num_scenarios = len(scenarios)
        x = cp.Variable(self.n_x)
        y = cp.Variable()
        mu = cp.Variable(num_scenarios, boolean=True)      # mu_j = 1 iff constraint satisfied for scenario j

        ## Cost
        cost = self.cost_func(x, y, scenarios, p_hat, E1, E2, E3, e, e_feas, num_scenarios, norm_type)

        ## Constraints
        constr = self.constraints(x, y, X, Y, mu, scenarios, p_hat, D1, D2, d, e_feas, eps, num_scenarios)
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(verbose=False, solver=self.SOLVER)
        if problem.status not in ["infeasible_or_unbounded", "infeasible", "unbonded"]:
            y_val = 0 if e_feas == 0 else y.value
            return cost.value, x.value, y_val, mu.value, problem.solver_stats.solve_time, problem.compilation_time, problem.status
        else:
            return None, None, None, None, None, None, None


class RobOptimizerLP(PartitionOptimizerLP):
    """Class to solve chance-constrained problem via robust approach (Margellos et al. 2015)"""

    def __init__(self, n_x, n_theta, SOLVER):
        super().__init__(n_x, n_theta, SOLVER)

    def constraints_robust(self, x, y, X, Y, scenarios, D1, D2, d, e_feas, slack):
        """Robust constraint"""
        K = 1
        constr = [X['A'] @ x <= X['b']]
        if e_feas:
            constr += [Y['A'] * y <= Y['b']]
        D1_vec = np.tile(D1, (K, 1))
        D2_vec = np.kron(np.eye(K), D2)
        scen_vec = scenarios.flatten()
        d_vec = np.tile(d, (1, K)).flatten()
        tau_vec = np.hstack([slack[j] for j in range(K)])
        constr += [D1_vec @ x + D2_vec @ scen_vec + d_vec + e_feas * y <= tau_vec]
        return constr

    def optimize_robust(self, X, Y, D1, D2, d, E1, E2, E3, e, e_feas, scenario, samples, norm_type=1, cost_type='avg'):
        """Solve robust optimization problem"""
        num_scenarios = 1
        x = cp.Variable(self.n_x)
        y = cp.Variable()
        slack = self.tau
        ## Cost
        if cost_type == 'avg':
            num_samples = len(samples)
            p_hat = np.ones(num_samples) * 1 / num_samples
            cost = self.cost_func(x, y, samples, p_hat, E1, E2, E3, e, e_feas, num_samples, norm_type)
        else:
            p_hat = [1]
            cost = self.cost_func(x, y, scenario, p_hat, E1, E2, E3, e, e_feas, num_scenarios, norm_type)

        ## Constraints
        constr = self.constraints_robust(x, y, X, Y, scenario, D1, D2, d, e_feas, slack)
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(verbose=False, solver=cp.GUROBI)
        if problem.status not in ["infeasible_or_unbounded", "infeasible", "unbonded"]:
            y_val = 0 if e_feas == 0 else y.value
            return cost.value, x.value, y_val, problem.solver_stats.solve_time, problem.compilation_time, problem.status
        else:
            return None, None, None, None, None, None, None



