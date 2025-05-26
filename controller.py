import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import polytope as pt
from scipy.special import comb
from itertools import combinations, product
from scipy.linalg import block_diag
import itertools
import time
import warnings


class OptimalControllerPWA:
    def __init__(self, n_x: int, n_u: int, n_unc: int, N_horizon: int,
                 modes: list, X: dict, U: dict,
                 Q: np.ndarray, R: np.ndarray,
                 x_ref: np.ndarray, u_ref: np.ndarray,
                 norm_type: int, eps: float, SOLVER: str):
        """
        Initialize OptimalControllerPWA with the simulations parameters.
        """

        self.n_x, self.n_u, self.n_unc, self.N_horizon = n_x, n_u, n_unc, N_horizon
        self.modes, self.num_modes_horizon = modes, len(modes) ** (self.N_horizon - 1)
        self.X, self.U = X, U
        self.Q, self.R = Q, R
        self.x_ref, self.u_ref = x_ref, u_ref
        self.norm_type, self.eps = norm_type, eps
        self.solver = SOLVER

        # Placeholders for compact matrices and solution details.
        self.F = self.G = self.L = None
        self.Xn = self.Un = self.Qn = self.Rn = None
        self.X_ref = self.U_ref = None
        self.u_optimal = self.x_optimal = self.cost_optimal = self.solver_time = None

    def compact_matrices_given_sequence(self, mode_seq: list, size_pwa_region: int):
        """
        Construct matrices for compact PWA dynamics:
        X+ = F_h x0 + G_h u + L_h \eta + v
        iff pwa_region_h['A'] X+ <= pwa_region_h['b']

        Args:
            mode_seq: A possible sequence of modes
            size_pwa_region: number of rows of pwa_region['A']

        Returns:
            Tuple of (F, G, L, V, pwa_region) defining PWA dynamics over horizon
        """

        N = self.N_horizon
        n_x, n_u, n_unc = self.n_x, self.n_u, self.n_unc

        # Build F and V for time steps 1,...,N
        F_blocks = []
        V_blocks = []
        current_F = np.eye(n_x)  # initially, F0 = I (x0 remains x0)
        current_V = np.zeros((n_x, 1))
        for k in range(N):
            mode_k = mode_seq[k]
            A_k = self.modes[mode_k]['A']
            v_k = self.modes[mode_k]['v']
            # Update matrices recursively and store them
            current_F = A_k @ current_F
            current_V = A_k @ current_V + v_k
            F_blocks.append(current_F)
            V_blocks.append(current_V)
        # Now, vertical stack from list.
        F_no_x0 = np.vstack(F_blocks)  # shape: (n_x * N, n_x)
        V_no_x0 = np.vstack(V_blocks)  # shape: (n_x * N, 1)

        # Append x0 on top (identity and 0 matrices)
        F_full = np.vstack([np.eye(n_x), F_no_x0])  # shape: (n_x*(N+1), n_x)
        V_full = np.vstack([np.zeros((n_x, 1)), V_no_x0])  # shape: (n_x*(N+1), 1)

        # Build G, L (from u and uncertainty to predicted trajectory)
        G_no_x0 = np.zeros((n_x * N, n_u * N))
        L_no_x0 = np.zeros((n_x * N, n_unc * N))
        for i in range(N):
            for j in range(i + 1):
                prod_G = np.eye(n_x)
                prod_L = np.eye(n_x)
                # Compute the product A_{i_i} ... A_{i_{j+1}}
                for k in range(j + 1, i + 1):
                    mode_k = mode_seq[k]
                    prod_G = self.modes[mode_k]['A'] @ prod_G
                    prod_L = self.modes[mode_k]['A'] @ prod_L
                mode_j = mode_seq[j]
                B_j = self.modes[mode_j]['B']
                C_j = self.modes[mode_j]['C']
                G_no_x0[i * n_x:(i + 1) * n_x, j * n_u:(j + 1) * n_u] = prod_G @ B_j
                L_no_x0[i * n_x:(i + 1) * n_x, j * n_unc:(j + 1) * n_unc] = prod_L @ C_j
        # Add one row on top for x0 (which does not depend on u or uncertainty).
        G_full = np.vstack([np.zeros((n_x, n_u * N)), G_no_x0])
        L_full = np.vstack([np.zeros((n_x, n_unc * N)), L_no_x0])

        # PWA dynamic constraints
        # At each time step k = 0,1,...,N-1, the constraint is E_{h_k} [x_k; u_k; eta_k] <= e_{h_k}.
        # We form the block diagonal matrix:
        E_blocks = []
        e_blocks = []
        for k in range(N):
            mode_k = mode_seq[k]
            E_k = self.modes[mode_k]['E']  # e.g., a matrix of size (n_constr_k, n_x+n_u+n_unc)
            e_k = self.modes[mode_k]['e']  # e.g., a vector of size (n_constr_k, 1)
            E_blocks.append(E_k)
            e_blocks.append(e_k)
        # Form block-diagonal matrix for all stages:
        E_comp = block_diag(*E_blocks)
        E_comp = np.hstack((E_comp, np.zeros((E_comp.shape[0], self.n_x))))
        e_comp = np.vstack(e_blocks)
        if E_comp.shape[0] < size_pwa_region:
            E_comp = np.vstack((E_comp, np.zeros((size_pwa_region - E_comp.shape[0], E_comp.shape[1]))))
            e_comp = np.vstack((e_comp, np.zeros((size_pwa_region - e_comp.shape[0], 1))))
        pwa_region = {'A': E_comp, 'b': e_comp}

        return F_full, G_full, L_full, V_full, pwa_region

    def get_all_compact_matrices(self, x0: np.ndarray) -> tuple:
        """
        Generate all compact matrices (F, G, L, V) for all feasible mode sequences starting at x0.

        Args:
            x0: Initial state (n_x by 1).

        Returns:
            Lists of F, G, L, V matrices and PWA regions for each mode sequence.
        """
        num_modes = len(self.modes)
        current_mode = 0
        for i in range(num_modes):
            if all(self.modes[i]['E'] @ x0 <= self.modes[i]['e']):
                current_mode = i
        # Generate all possible mode sequences for the horizon.
        mode_options = [i for i in range(num_modes)]
        dim_initial_mode = self.modes[current_mode]['E'].shape[0]
        max_dim = max([self.modes[i]['E'].shape[0] for i in range(num_modes)])
        max_dim_over_horizon = dim_initial_mode + max_dim * (self.N_horizon - 1)
        all_mode_seqs = list(itertools.product(mode_options, repeat=self.N_horizon))
        filtered_seqs = [seq for seq in all_mode_seqs if seq[0] == current_mode]
        all_results = [self.compact_matrices_given_sequence(mode_seq, max_dim_over_horizon)
                       for mode_seq in filtered_seqs]
        F_list, G_list, L_list, V_list, pwa_region = zip(*all_results)

        return list(F_list), list(G_list), list(L_list), list(V_list), list(pwa_region)

    def set_compact_matrices(self, x0: np.ndarray) -> None:
        """
        Set compact matrices and related PWA region constraints from all feasible mode sequences from x0.

        Args:
            x0: Initial state (n_x , 1).
        """
        self.F, self.G, self.L, self.V, self.pwa_regions_horizon = self.get_all_compact_matrices(x0)
        Hx = np.kron(np.eye(self.N_horizon), self.X['A'])
        Hx = np.hstack((np.zeros((Hx.shape[0], self.n_x)), Hx))
        Hu = np.kron(np.eye(self.N_horizon), self.U['A'])
        hx = np.tile(self.X['b'], (self.N_horizon, 1))
        hu = np.tile(self.U['b'], (self.N_horizon, 1))
        Xn, Un = {}, {}
        Xn['A'] = Hx
        Xn['b'] = hx
        Un['A'] = Hu
        Un['b'] = hu
        self.Xn = Xn
        self.Un = Un
        self.Qn = np.kron(np.eye(self.N_horizon), self.Q)
        self.Rn = np.kron(np.eye(self.N_horizon), self.R)
        self.X_ref = np.tile(self.x_ref, (self.N_horizon, 1))
        self.U_ref = np.tile(self.u_ref, (self.N_horizon, 1))

    def compute_sampled_pwa_dynamics_d(self, x0: np.ndarray, u_seq: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """
        Given x0, a sequence of input, and some uncertainty samples, computes the possible PWA dynamics.

        Args:
            x0: Initial state (n_x , 1).
            u_seq: Control sequence (n_u * N_horizon , 1).
            samples: Uncertainty samples (Nsamples , n_unc * N_horizon).

        Returns:
            Array of samples PWA dynamics.
        """
        X = []
        Nsamples = len(samples)
        # Compute x_test for all scenarios
        for h in range(self.num_modes_horizon):
            region = self.pwa_regions_horizon[h]
            x_h = self.F[h] @ x0 + self.G[h] @ u_seq
            x_test_all = np.tile(x_h, (1, Nsamples)) + self.L[h] @ samples.T
            condition_pwa_dyn = np.all(region['A'] @ x_test_all <= region['b'], axis=0)
            x_test_in_region_h = x_test_all[:, condition_pwa_dyn]
            num_samples_region_h = x_test_in_region_h.shape[1]
            if num_samples_region_h >= 1:
                X.append(x_test_in_region_h.transpose())
        return np.vstack(X)

    def compute_sampled_pwa_dynamics_u(self, x0: np.ndarray, u_samples: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Given x0, a fixed value for uncertainty, and some input samples, computes the possible PWA dyncamis.

        Args:
            x0: Initial state (n_x , 1).
            u_samples: Sampled control sequences (Nsamples , n_u * N_horizon).
            d: Fixed uncertainty vector (n_unc * N_horizon , 1).

        Returns:
            Array of sampled PWA dynamics.
        """
        X = []
        Nsamples = len(u_samples)
        # Compute x_test for all input scenarios
        for h in range(self.num_modes_horizon):
            region = self.pwa_regions_horizon[h]
            x_h = self.F[h] @ x0 + self.L[h] @ d
            x_test_all = np.tile(x_h, (1, Nsamples)) +self.G[h] @ u_samples.T
            condition_pwa_dyn = np.all(region['A'] @ x_test_all <= region['b'], axis=0)
            x_test_in_region_h = x_test_all[:, condition_pwa_dyn]
            num_samples_region_h = x_test_in_region_h.shape[1]
            if num_samples_region_h >= 1:
                X.append(x_test_in_region_h.transpose())
        return np.vstack(X)

    def get_empirical_violation(self, x0: np.ndarray, u_seq: np.ndarray, samples: np.ndarray) -> float:
        """
        Given x0, a sequence of input, and some uncertainty samples, computes the empirical constraint violation

        Args:
            x0: Initial state.
            u_seq: Control sequence.
            samples: Matrix of disturbance samples.

        Returns:
            Violation rate (float in [0, 1]).
        """
        X = self.compute_sampled_pwa_dynamics_d(x0, u_seq, samples)
        # Note: since self.Xn['b'] is a column vector, we take just the row in it (and check along the columns)
        satisfaction = np.sum(np.all(X @ self.Xn['A'].transpose() <= self.Xn['b'].transpose()[0] + 1e-9, axis=1)) / len(samples)
        return 1 - satisfaction

    def get_worst_case(self, x0: np.ndarray, u_seq: np.ndarray, samples: np.ndarray) -> int:
        """
        Get the time step where the predicted trajectories are closest to the constraint boundaries.

        Args:
            x0: Initial state.
            u_seq: Control sequence.
            samples: Disturbance samples.

        Returns:
            Index of the time step with worst-case constraint margin.
        """
        # Compute all sample trajectories
        X = self.compute_sampled_pwa_dynamics_d(x0, u_seq, samples)   # shape (Nsamples, n_x*(N_horizon+1))
        # Check distance from boundary (dist should be <= 0 for entries that satisfy constraint)
        # Note constraints holds from time 1 to N_horizon
        dist = X @ self.Xn['A'].transpose() - self.Xn['b'].transpose()[0]   # shape (Nsamples, n_constr*(N_horizon))
        n_constr_single = self.X['b'].size
        # For each time step, compute max over all scenarios
        max_over_entry = np.max(dist, axis=0)   # shape (n_x*(N_horizon),)
        idx_max_over_entry = np.argmax(max_over_entry)
        # The time step of the uncertainty which should be splitted (note that state at next step would be time_max_state_constr + 1)
        time_to_split = int(idx_max_over_entry / n_constr_single)
        return time_to_split


class PartitionControllerPwa(OptimalControllerPWA):
    def __init__(self, n_x: int, n_u: int, n_unc: int, N_horizon: int,
                modes: list, X: dict, U: dict,
                 Q: np.ndarray, R: np.ndarray,
                 x_ref: np.ndarray, u_ref: np.ndarray,
                 norm_type: int, eps: float, SOLVER: str):
        """
        Initialize PartitionControllerPwa with inherited and additional attributes for partition-based tightening.
        """
        super().__init__(n_x, n_u, n_unc, N_horizon, modes, X, U, Q, R, x_ref, u_ref, norm_type, eps, SOLVER)
        self.V = None
        self.all_compact_dynamics = None

        self.pwa_regions_horizon = None
        self.tightening = self.gamma = None
        self.min_smallest_sv = None
        self.lip_u = self.lip_unc = None
        self.cost_ub = self.cost_lb_aux = self.cost_lb_th = None
        self.u_feas = None
        self.solver = SOLVER

    def optimize(self, x0: np.ndarray, delta: float, p_hat: np.ndarray, scenarios: np.ndarray,
                 tight_or_relax: str) -> tuple:
        """
        Solve the partition-based optimization problem (with tightening or relaxation)

        Args:
            x0: Initial state.
            delta: tightening/relaxation.
            p_hat: Clusters probabilities.
            scenarios: Representative scenarios.
            tight_or_relax: 'tight' or 'relax' flag.

        Returns:
            Tuple with optimal cost, control, binary variables, solve time, and solver status.
        """
        warnings.simplefilter("ignore")
        Nscen = len(scenarios)
        U = cp.Variable((self.n_u * self.N_horizon, 1))
        mu = cp.Variable((Nscen, self.num_modes_horizon),
                         boolean=True)  # mu_{j, h} = 1 if mode h robustly satisfies scenario j
        alpha = cp.Variable((Nscen, self.num_modes_horizon), boolean=True)  # alpha_{j, h} = 1 iff scenario j triggers mode h
        # Tightening
        if tight_or_relax == 'tight':
            constraint_modification = self.tightening
            eps_modification = - delta
            constraint_modification = np.squeeze(constraint_modification, axis=-1)
        elif tight_or_relax == 'relax':
            constraint_modification = self.gamma
            eps_modification = delta
            constraint_modification = np.squeeze(constraint_modification, axis=-1)
        else:
            dim_modification = self.Xn['b'].shape[0] + self.pwa_regions_horizon[0]['b'].shape[0]
            constraint_modification = np.zeros((Nscen, self.num_modes_horizon, dim_modification))
            eps_modification = 0
        # Init cost and constraints
        constr = [self.Un['A'] @ U <= self.Un['b']]  # Input constraints
        cost = cp.norm(self.Rn @ (U - self.U_ref), self.norm_type)
        # Write scenario trajectory constraint
        constr += [cp.sum(alpha, axis=1) == 1]  # each scenario triggers exactly on mode
        # For each mode h, build the constraints for all scenarios at once.
        M = 100  # Big-M constant
        # Y is a list, in a way that Y[h] are all the scenario trajectories given mode h
        state_dim = (self.N_horizon + 1) * self.n_x
        Y_all = cp.Variable((Nscen * self.num_modes_horizon, state_dim))
        Y_all_reshaped = cp.reshape(Y_all, (Nscen, self.num_modes_horizon, state_dim), order='F')
        Xscen = cp.sum(Y_all_reshaped, axis=1)  # shape: (Nscen, state_dim)

        # Build the Big-M dynamics constraints for Y_all
        # Each scenario trajectory is a row vector, then stacked vertically for all modes
        dynamics_list = [cp.transpose(self.F[h] @ x0 + self.G[h] @ U) + scenarios @ self.L[h].transpose()
            for h in range(self.num_modes_horizon)]
        dynamics_all = cp.vstack(dynamics_list)  # shape: (Nscen*num_modes, state_dim)
        # Similarly, for each mode we want the corresponding alpha and mu to be stacked in the same order.
        alpha_all = cp.vstack([cp.reshape(alpha[:, h], (Nscen, 1), order='F') for h in range(self.num_modes_horizon)])
        # Now, add the big-M constraints on Y_all:
        constr += [
            Y_all <= M * alpha_all,
            Y_all >= -M * alpha_all,
            Y_all <= dynamics_all + M * (1 - alpha_all),
            Y_all >= dynamics_all - M * (1 - alpha_all)
        ]

        # Nominal trajectory constraints
        # For each mode h, the nominal PWA region constraint is:
        # A_region @ Xscen[j] <= b_region + 1000*(1 - alpha[j, h]) for all scenarios j.
        # Stack left-hand and right-hand side for all modes
        nominal_lhs = cp.vstack([
            cp.matmul(Xscen, self.pwa_regions_horizon[h]['A'].T)
            for h in range(self.num_modes_horizon)
        ])

        nominal_rhs = cp.vstack([
            self.pwa_regions_horizon[h]['b'].T + 1000 * (1 - cp.reshape(alpha[:, h], (Nscen, 1), order='F'))
            for h in range(self.num_modes_horizon)
        ])

        constr.append(nominal_lhs <= nominal_rhs)

        # Robust constraints
        # For each mode h, the robust constraint is:
        #   (A_total) @ Xscen[j] <= (b_total)^T + constraint_modification[j, h] + 1000*(1 - mu[j, h])
        # where:
        #   A_total = np.vstack((self.Xn['A'], self.pwa_regions_horizon[h]['A']))
        #   b_total = np.vstack((self.Xn['b'], self.pwa_regions_horizon[h]['b']))
        # Stack lhs and rhs for all modes
        robust_lhs = cp.vstack([
            cp.matmul(Xscen, np.vstack((self.Xn['A'], self.pwa_regions_horizon[h]['A'])).T)
            for h in range(self.num_modes_horizon)
        ])
        robust_rhs = cp.vstack([
            np.vstack((self.Xn['b'], self.pwa_regions_horizon[h]['b'])).T +
            cp.reshape(constraint_modification[:, h], (Nscen, -1), order='F') +
            1000 * (1 - cp.reshape(mu[:, h], (Nscen, 1), order='F'))
            for h in range(self.num_modes_horizon)
        ])
        constr.append(robust_lhs <= robust_rhs)
        # Chance constraint
        # Sum mu[j, h]*p_hat[j] over all scenarios and modes:
        sum_satisfaction = cp.sum(cp.multiply(mu, cp.reshape(p_hat, (Nscen, 1), order='F')))
        constr += [sum_satisfaction >= 1 - (self.eps + eps_modification)]

        # Add state cost
        diff_cost = Xscen[:, self.n_x:] - self.X_ref.T
        diff_cost = self.Qn @ diff_cost.T
        scen_cost = cp.sum(cp.abs(diff_cost), axis=0)
        cost += cp.sum(cp.multiply(p_hat, scen_cost))

        # Solve with cvxpy
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(verbose=False, solver=self.solver)
        if tight_or_relax == 'tight':
            self.u_feas = U.value
        self.u_optimal = U.value
        self.cost_optimal = cost.value
        self.solver_time = problem.solver_stats.solve_time
        if problem.status not in ["infeasible_or_unbounded", "infeasible", "unbonded"]:
            return cost.value, U.value, mu.value, alpha.value, problem.solver_stats.solve_time, problem.status
        else:
            return None, None, None, None, None, problem.status

    def _compute_correction_general(self, scenarios: np.ndarray, regions: list, correction: str) -> list:
        """
        Compute either tightening or relaxation, for general polyhedral uncertainty sets.

        Args:
            scenarios: Representative scenarios.
            regions: Polyhedral uncertainty regions (with 'A' and 'b').
            correction: 'tightening' or 'relaxation'.

        Returns:
            List of lists of correction vectors (one per mode and cluster).
        """
        N_scen = len(regions)
        n_theta = self.N_horizon * self.n_unc
        corrected_values = [[] for _ in range(N_scen)]
        for j in range(N_scen):
            uncertainty_region, scenario = regions[j], scenarios[j]
            theta_hat = scenario.reshape(-1, 1)
            # Build the constraint: kron(I, A_region) * vec(theta) <= tile(b_region)
            for h in range(self.num_modes_horizon):
                D = np.vstack((self.Xn['A'] @ self.L[h], self.pwa_regions_horizon[h]['A'] @ self.L[h]))
                dim = D.shape[0]
                theta = cp.Variable((n_theta, dim))
                constr = [np.kron(np.eye(dim), uncertainty_region['A']) @ cp.vec(theta, order='F').reshape((-1, 1), 'C')
                          <= np.tile(uncertainty_region['b'], (dim, 1))]
                # Compute the difference; note that reversing the order flips the sign.
                if correction == 'tightening':
                    # For tightening: theta - tile(theta_hat)
                    theta_matrix = theta - np.tile(theta_hat, (1, dim))
                else:
                    # For gamma: tile(theta_hat) - theta
                    theta_matrix = np.tile(theta_hat, (1, dim)) - theta

                cost = cp.sum(cp.multiply(D, theta_matrix.T))
                problem = cp.Problem(cp.Maximize(cost), constr)
                problem.solve(verbose=False, solver='GUROBI')

                # Compute the dot product for each row of D with the corresponding column of theta_matrix.
                value = np.array([np.dot(D[i, :], theta_matrix.value[:, i])
                                  for i in range(dim)]).reshape(-1, 1)
                # For tightening we want the negative of this value.
                corrected_values[j].append(-value if correction == 'tightening' else value)
        return corrected_values

    def _compute_correction_box(self, scenarios: np.ndarray, boxes: list, correction: str) -> list:
        """
        Compute constraint tightening/relaxation for box-shaped uncertainty sets.

        Args:
            scenarios: Representative scenarios.
            boxes (list): A list of box uncertainty sets, as dictionaries.
            correction: Constraint correction (either tightening or relaxation)

        Returns:
            List of lists of correction vectors (one per mode and cluster).
        """
        N_scen = len(scenarios)
        corrected_values = [[] for _ in range(N_scen)]
        for j in range(N_scen):
            box_j = boxes[j].T
            lb_j, ub_j = box_j[0], box_j[1]
            scenario = scenarios[j]
            if correction == 'tightening':
                lb_j_trans = lb_j - scenario
                ub_j_trans = ub_j - scenario
            else:
                lb_j_trans = scenario - ub_j
                ub_j_trans = scenario - lb_j
            for h in range(self.num_modes_horizon):
                D = np.vstack((self.Xn['A'] @ self.L[h], self.pwa_regions_horizon[h]['A'] @ self.L[h]))
                # Select ub where condition is True and lb otherwise.
                selected_values = np.where(D >= 0, ub_j_trans, lb_j_trans)
                # Multiply elementwise by A and sum along each row (axis=1).
                result = np.sum(D * selected_values, axis=1)
                value = result.reshape(-1, 1)
                # For tightening we want the negative of this value.
                corrected_values[j].append(-value if correction == 'tightening' else value)
        return corrected_values

    def set_tightening(self, scenarios: np.ndarray, regions: list = None, boxes: list = None) -> None:
        """
        Set constraint tightening from box or polyhedral regions.

        Args:
            scenarios: Representative scenarios.
            regions: Polyhedral sets.
            boxes: Box sets.
        """
        if boxes is not None:
            self.tightening = self._compute_correction_box(scenarios, boxes, 'tightening')
        elif regions is not None:
            self.tightening = self._compute_correction_general(scenarios, regions, 'tightening')
        else:
            raise ValueError('Either specify polytopes or boxes for uncertainty sets.')

    def set_gamma(self, scenarios: np.ndarray, regions: list = None, boxes: list = None) -> None:
        """
        Set constraint relaxation from box or polyhedral regions.

        Args:
            scenarios: Representative scenarios.
            regions: Polyhedral sets.
            boxes: Box sets.
        """
        if boxes is not None:
            self.gamma = self._compute_correction_box(scenarios, boxes, 'relaxation')
        elif regions is not None:
            self.gamma = self._compute_correction_general(scenarios, regions, 'relaxation')
        else:
            raise ValueError('Either specify polytopes or boxes for uncertainty sets.')

    def set_minimum_smallest_sv(self) -> None:
        """
        Compute the minimum smallest singular value across all invertible square submatrices,
        of matrix 'A' of the polytope defining the intersection between constraint and pwa region.
        """
        min_smallest_sv = []
        for h in range(self.num_modes_horizon):
            # Wrt linear case, have to compute intersection with region
            Xn_and_region = np.vstack((self.Xn['A'], self.pwa_regions_horizon[h]['A']))
            A = Xn_and_region @ self.G[h]
            m, n = A.shape
            if m < n:
                raise ValueError("Matrix must have at least as many rows as columns to form square submatrices.")
            # Generate all combinations of rows to form square submatrices
            row_indices = range(m)
            sv_list = []
            for rows in combinations(row_indices, n):
                # Extract the square submatrix
                submatrix = A[np.array(rows), :]
                # Check if it's invertible (determinant non-zero)
                if np.abs(np.linalg.det(submatrix)) >= 1e-9:
                    min_sv = np.min(np.linalg.svd(submatrix, compute_uv=False))
                    # min_sv = 1 / np.linalg.norm(np.linalg.inv(submatrix), 2)
                    sv_list.append(min_sv)
            min_smallest_sv.append(min(sv_list))
        self.min_smallest_sv = min(min_smallest_sv)

    def set_lip_constant_u(self) -> float:
        """
        Estimate Lipschitz constant of cost function w.r.t. control input for 1-norm cost.

        Returns:
        Lipschitz constant (float).
        """
        if self.norm_type == 1:
            Qn = self.Qn
            Rn = self.Rn
            L1 = np.linalg.norm(Qn, 1)
            L2_list = []
            for h in range(self.num_modes_horizon):
                L2_list.append(np.linalg.norm(self.G[h], 1))
            self.lip_u = (L1 * max(L2_list) + np.linalg.norm(Rn, 1)) * np.sqrt(self.n_u * self.N_horizon)
            return self.lip_u
        else:
            raise ValueError('So far, only 1-norm for PWA performance bounds!')

    def set_lip_constant_unc(self) -> float:
        """
        Estimate Lipschitz constant of cost function w.r.t. uncertainty for 1-norm cost.
        Returns:
        Lipschitz constant (float).
        """
        if self.norm_type == 1:
            Qn = self.Qn
            L1 = np.linalg.norm(Qn, 1)
            L2_list = []
            for h in range(self.num_modes_horizon):
                L2_list.append(np.linalg.norm(self.L[h], 1))
            self.lip_unc = L1 * max(L2_list)
            return self.lip_unc
        else:
            raise ValueError('So far, only 1-norm for PWA performance bounds!')

    def set_lip_constant_unc_sampling(self, x0: np.ndarray, uncertainty_box: np.ndarray) -> float:
        """
        Empirically estimate Lipschitz constant w.r.t. uncertainty via sampling.

        Args:
            x0: Initial state.
            uncertainty_box: Box describing uncertainty domain.

        Returns:
            Empirical Lipschitz constant (float).
        """
        if self.norm_type == 1:
            Qn = self.Qn
            Rn = self.Rn
            uncertainty_box = uncertainty_box.T
            dim = uncertainty_box.shape[1]
            samples1 = np.random.uniform(low=uncertainty_box[0], high=uncertainty_box[1], size=(500, dim))
            samples2 = np.random.uniform(low=uncertainty_box[0], high=uncertainty_box[1], size=(500, dim))
            poly_Un = pt.Polytope(self.Un['A'], self.Un['b'])
            vertices_u = pt.extreme(poly_Un)
            lip = []
            for u in vertices_u:
                u = u.reshape(-1, 1)
                samples_dyn_1 = self.compute_sampled_pwa_dynamics(x0, u, samples1)
                samples_dyn_2 = self.compute_sampled_pwa_dynamics(x0, u, samples2)
                cost_1 = np.linalg.norm((samples_dyn_1[:, self.n_x:] - self.X_ref[:, 0]) @ Qn, ord=1, axis=1)
                cost_2 = np.linalg.norm((samples_dyn_2[:, self.n_x:] - self.X_ref[:, 0]) @ Qn, ord=1, axis=1)
                num = cost_1[:, None] - cost_2[None, :]
                den = np.abs(samples1[:, None, :] - samples2[None, :, :]).sum(axis=-1)
                lip.append(np.max(num / den))
            empirical_lip_constant = max(lip)
            self.lip_unc = empirical_lip_constant
            return self.lip_unc
        else:
            raise ValueError('So far, only 1-norm for PWA performance bounds!')

    def set_lip_constant_u_sampling(self, x0: np.ndarray, uncertainty_box: np.ndarray) -> float:
        """
        Empirically estimate Lipschitz constant w.r.t. control input via sampling.

        Args:
            x0: Initial state.
            uncertainty_box: Box describing uncertainty support.

        Returns:
            Empirical Lipschitz constant (float).
        """
        if self.norm_type == 1:
            Qn = self.Qn
            Rn = self.Rn
            u_box = np.array([-np.ones(self.N_horizon * self.n_u), np.ones(self.N_horizon * self.n_u)]) * np.abs(
                self.U['b'][0, 0])
            dim = u_box.shape[1]
            samples1 = np.random.uniform(low=u_box[0], high=u_box[1], size=(500, dim))
            samples2 = np.random.uniform(low=u_box[0], high=u_box[1], size=(500, dim))
            poly_unc = pt.box2poly(uncertainty_box)
            vertices_unc = pt.extreme(poly_unc)
            lip = []
            for d in vertices_unc:
                d = d.reshape(-1, 1)
                samples_dyn_1 = self.compute_sampled_pwa_dynamics_u(x0, samples1, d)
                samples_dyn_2 = self.compute_sampled_pwa_dynamics_u(x0, samples2, d)
                cost_1 = np.linalg.norm((samples_dyn_1[:, self.n_x:] - self.X_ref[:, 0]) @ Qn, ord=1, axis=1) \
                         + np.linalg.norm(samples1 @ Rn)
                cost_2 = np.linalg.norm((samples_dyn_2[:, self.n_x:] - self.X_ref[:, 0]) @ Qn, ord=1, axis=1) \
                         + np.linalg.norm(samples2 @ Rn)
                num = cost_1[:, None] - cost_2[None, :]
                den = np.abs(samples1[:, None, :] - samples2[None, :, :]).sum(axis=-1)
                lip.append(np.max(num / den))
            empirical_lip_constant = max(lip) * np.sqrt(self.n_u * self.N_horizon)
            self.lip_u = empirical_lip_constant
            return self.lip_u
        else:
            raise ValueError('So far, only 1-norm for PWA performance bounds!')

    def compute_c(self, nominal_scen: list, clusters: list, N_samples: int) -> float:
        """
        Compute error term due to partitioning.

        Args:
            nominal_scen: List of representative scenarios.
            clusters: List of clusters.
            N_samples: Number of samples.

        Returns:
            Error term (float).
        """
        c = 0
        for j in range(len(clusters)):
            Theta_hat = np.tile(nominal_scen[j], (len(clusters[j]), 1))
            c += np.sum(np.abs((Theta_hat - clusters[j])))
        c = c / N_samples
        c = c * self.lip_unc
        return c

    def get_performances(self, x0, lam, clusters, nominal_scen, p_hat, delta, Nsamples, validation_set, lb, r):
        """
        Get controller performance: upper and lower bound, and empirical probability of constraint violation,
        associated to approximate control problem.

        Args:
            x0: Initial state.
            lam: Empirical error in cost function approximtion.
            clusters: Scenario clusters.
            nominal_scen: Nominal scenarios.
            p_hat: Cluster probabilities.
            delta: Empirical error in probabilities approximation.
            Nsamples: Number of samples.
            validation_set: Sample validation set, to compute empirical probability of violation.
            lb: Type of lower bound to compute: 'th', 'aux', 'both'.
            r: Sampling radius.

        Returns:
            Tuple of (empirical violation, lower bound, upper bound, total solver time).
        """
        cost_pp, u_feas, _, _, time_ub, _ = self.optimize(x0, delta, p_hat, nominal_scen, 'tight')
        if u_feas is None:
            return -1, None, None, None, None
        else:
            empirical_violation = self.get_empirical_violation(x0, u_feas, validation_set)
            c = self.compute_c(nominal_scen, clusters, Nsamples)
            ub = cost_pp + c + lam + 2 * self.lip_u * r
            if lb == 'th':
                bound_lhs = self.lip_u * (np.max(self.gamma) - np.max(self.tightening)) * np.sqrt(
                    self.n_u * self.N_horizon) / self.min_smallest_sv + c + lam
                lb_th = cost_pp - bound_lhs
                return empirical_violation, lb_th, ub
            elif lb == 'aux':
                cost_lb, _, _, _, time_lb_aux, _ = self.optimize(x0, delta, p_hat, nominal_scen, 'relax')
                lb_aux = cost_lb - c - lam
                return empirical_violation, lb_aux, ub
            elif lb == 'both':
                bound_lhs = self.lip_u * (np.max(np.array(self.gamma) - np.array(self.tightening))) * np.sqrt(
                    self.n_u * self.N_horizon) / self.min_smallest_sv + c + lam
                lb_th = cost_pp - bound_lhs
                cost_lb_aux, _, _, _, time_lb_aux, _ = self.optimize(x0, delta, p_hat, nominal_scen, 'relax')
                lb_aux = cost_lb_aux - c - lam - 2 * self.lip_u * r
                return empirical_violation, lb_th, lb_aux, ub, time_ub + time_lb_aux
            else:
                return empirical_violation, None, None, ub, time_ub


class RandomizedController(OptimalControllerPWA):
    def __init__(self, n_x: int, n_u: int, n_unc: int, N_horizon: int,
                 modes: list, X: dict, U: dict, Q: np.ndarray, R: np.ndarray,
                 x_ref: np.ndarray, u_ref: np.ndarray, norm_type: int, eps: float,
                 SOLVER: str):
        """
        Initialize RandomizedController with inherited structure.
        """
        super().__init__(n_x, n_u, n_unc, N_horizon, modes, X, U, Q, R, x_ref, u_ref, norm_type, eps,
                         SOLVER)

    def optimize(self, x0: np.ndarray, scenarios: np.ndarray) -> tuple:
        """
        Solve a randomized optimization problem given N samples.

        Args:
            x0: Initial state (n_x , 1).
            scenarios: Sampled uncertainty scenarios (N_scenarios , (n_unc * N_horizon)).

        Returns:
            Tuple of (optimal cost, optimal input, mode-scenario assignment, solve time, status),
            or None values if problem is infeasible.
        """
        num_modes_horizon = int(self.num_modes_horizon)
        Nscen = len(scenarios)
        U = cp.Variable((self.n_u * self.N_horizon, 1))
        alpha = cp.Variable((Nscen, num_modes_horizon), boolean=True)  # alpha_{j, h} = 1 iff scenario j triggers mode h
        # Init cost and constraints
        constr = [self.Un['A'] @ U <= self.Un['b']]  # Input constraints
        cost = cp.norm(self.Rn @ (U - self.U_ref), self.norm_type)
        # Write scenario trajectory constraint
        constr += [cp.sum(alpha, axis=1) == 1]  # each scenario triggers exactly on mode
        # For each mode h, build the constraints for all scenarios at once.
        M = 1000  # Big-M constant
        # Y is a list, in a way that Y[h] are all the scenario trajectories given mode h
        state_dim = (self.N_horizon + 1) * self.n_x
        num_modes = num_modes_horizon
        Y_all = cp.Variable((Nscen * num_modes, state_dim))
        Y_all_reshaped = cp.reshape(Y_all, (Nscen, num_modes, state_dim), order='F')
        Xscen = cp.sum(Y_all_reshaped, axis=1)  # shape: (Nscen, state_dim)

        # Build the Big-M dynamics constraints for Y_all
        # Each scenario trajectory is a row vector, then stacked vertically for all modes
        dynamics_list = [cp.transpose(self.F[h] @ x0 + self.G[h] @ U) + scenarios @ self.L[h].transpose()
            for h in range(num_modes)]
        dynamics_all = cp.vstack(dynamics_list)  # shape: (Nscen*num_modes, state_dim)
        # Similarly, for each mode we want the corresponding alpha and mu to be stacked in the same order.
        alpha_all = cp.vstack([cp.reshape(alpha[:, h], (Nscen, 1), order='F') for h in range(num_modes)])
        # Now, add the big-M constraints on Y_all:
        constr += [
            Y_all <= M * alpha_all,
            Y_all >= -M * alpha_all,
            Y_all <= dynamics_all + M * (1 - alpha_all),
            Y_all >= dynamics_all - M * (1 - alpha_all)
        ]

        # Nominal trajectory constraints
        # For each mode h, the nominal PWA region constraint is:
        # A_region @ Xscen[j] <= b_region + 1000*(1 - alpha[j, h]) for all scenarios j.
        # Stack left-hand and right-hand side for all modes
        nominal_lhs = cp.vstack([
            cp.matmul(Xscen, self.pwa_regions_horizon[h]['A'].T)
            for h in range(num_modes)
        ])

        nominal_rhs = cp.vstack([
            self.pwa_regions_horizon[h]['b'].T + 1000 * (1 - cp.reshape(alpha[:, h], (Nscen, 1), order='F'))
            for h in range(num_modes)
        ])
        constr.append(nominal_lhs <= nominal_rhs)

        # Nominal state constraints
        # For each mode h, the nominal PWA region constraint is:
        # A_region @ Xscen[j] <= b_region + 1000*(1 - alpha[j, h]) for all scenarios j.
        # Stack left-hand and right-hand side for all modes
        state_lhs = cp.matmul(Xscen, self.Xn['A'].T)

        state_rhs = self.Xn['b'].T
        constr.append(state_lhs <= state_rhs)

        # Add state cost
        diff_cost = Xscen[:, self.n_x:] - self.X_ref.T
        diff_cost = self.Qn @ diff_cost.T
        scen_cost = cp.sum(cp.abs(diff_cost), axis=0)
        cost += cp.sum(scen_cost) * 1 / Nscen

        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(verbose=False, solver='GUROBI')
        self.u_feas = U.value
        self.u_optimal = U.value
        self.cost_optimal = cost.value
        self.solver_time = problem.solver_stats.solve_time
        if problem.status not in ["infeasible_or_unbounded", "infeasible", "unbonded"]:
            return cost.value, U.value, alpha.value, problem.solver_stats.solve_time, problem.status
        else:
            return None, None, None, None, problem.status

    def solve_binomial_sum(self, d: int, epsilon: float, beta: float, max_iter: int = 1000) -> int:
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
        def binomial_sum(N):
            """Computes the summation for a given N."""
            return sum(comb(N, i) * (epsilon ** i) * ((1 - epsilon) ** (N - i)) for i in range(d)) * self.num_modes_horizon
        # Initial bounds for bisection
        high = 2 / epsilon * (d - 1 + np.log(1 / beta))
        N_ini = round(high / 4)
        # Bisection method
        for _ in range(max_iter):
            beta_eval = binomial_sum(N_ini)
            if beta_eval <= beta:
                return N_ini
            N_ini += 1
        raise ValueError('Insufficient number of iterations.')

