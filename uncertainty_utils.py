import numpy as np
import matplotlib.pyplot as plt
import polytope as pt
from sklearn.cluster import KMeans
from itertools import product
import warnings
import matplotlib
# matplotlib.use("MacOSX")

class BasicUncertainty:
    """
    Base class for uncertainty handling.

    Attributes:
        n_uncertainty (int): Number of uncertainty dimensions.
        domain: Polytope representing the uncertainty domain.
    """


    def __init__(self, n_uncertainty: int):
        self.n_uncertainty = n_uncertainty
        self.domain = None

    def get_samples(self, Nsamples:int, seed: int) -> np.ndarray:
        pass

    def get_samples_batches(self, num_batches:int, Nsamples:int) -> np.ndarray:
        """
        Get batches of samples for multiple experiments
        """
        batches = []
        for i in range(num_batches):
            batches.append(self.get_samples(Nsamples, seed=i+1000))
        return batches

    def my_trunc_norm(self, means:np.ndarray, stdevs:np.ndarray, size:int, low:np.ndarray, high:np.ndarray,
                      N:int, seed:int=1, n_batches:int=1) -> np.ndarray:
        """
        Samples from a truncated normal distribution.
        """

        rng = np.random.default_rng(seed)

        total_needed = n_batches * N
        num_to_resample = total_needed
        samples_ok = []

        while num_to_resample > 0:
            samples = rng.normal(
                means,
                stdevs,
                size=(num_to_resample, size),
            )

            mask = (samples >= low) & (samples <= high)
            rows_ok = mask.all(axis=1)

            accepted = samples[rows_ok]
            num_ok = accepted.shape[0]

            num_to_resample -= num_ok
            samples_ok.append(accepted)

        samples_to_return = np.vstack(samples_ok)[:total_needed]
        samples_to_return = samples_to_return.reshape(n_batches, N, size)

        return samples_to_return


    def plot_samples(self, samples: np.ndarray, idx_to_plot: list, centers=None) -> None:
        """
        Plot sample trajectories for uncertainty components in idx_to_plot (max 2).

        Args:
            samples (np.ndarray): A list or array of sample vectors.
            idx_to_plot: list of coordinates to plot (max 2)
        """

        fig, axes = plt.subplots(1, 1, figsize=(8, 6))

        axes.plot(samples[:, idx_to_plot[0]], samples[:, idx_to_plot[1]], color='blue', marker='o', linestyle='None', markersize=2)
        axes.set_xlabel(r"$\theta_" + str(idx_to_plot[0]) + r"$")
        axes.set_ylabel(r"$\theta_" + str(idx_to_plot[1]) + r"$")
        axes.grid(True)

        if centers is not None:
            axes.plot(centers[:, idx_to_plot[0]], centers[:, idx_to_plot[1]], color='red', marker='x',
                      linestyle='None', markersize=10)

        plt.tight_layout()
        plt.show()

    def set_domain(self, lb: np.ndarray, ub: np.ndarray) -> None:
        """
        Set the uncertainty domain as a polytope.

        Args:
            lb (np.ndarray): Lower bounds.
            ub (np.ndarray): Upper bounds.
        """
        self.domain = pt.Polytope(lb, ub)

    def my_kmeans_partition(self, samples: np.ndarray, K: int, initial_set, dim: int, random_state: int=0) -> list:
        """
        Partition the sample space using KMeans clustering and a Voronoi partitioning.

        Args:
            samples (np.ndarray): Samples to cluster/partition
            K (int): Number of clusters.
            initial_set: Initial polytope (must have attributes A and b).
            dim (int): Dimensionality of the space.

        Returns:
            list: A list of regions defined by hyperplanes.
        """
        clustering_algo = KMeans(n_clusters=K, init='k-means++', random_state=random_state)
        clustering_algo.fit(samples)
        centers = clustering_algo.cluster_centers_
        regions = self.my_voronoi(K, centers, initial_set, dim)
        return regions

    def my_voronoi(self, K: int, centers: np.ndarray, initial_set, dim: int) -> list:
        """
        Construct a Voronoi partition of the space using hyperplanes.

        Args:
            K (int): Number of clusters.
            centers (np.ndarray): Cluster centers.
            initial_set: Initial polytope (with attributes A and b).
            dim (int): Dimensionality.

        Returns:
            list: A list of regions represented as dicts with keys 'A' and 'b'.
        """
        # Compute hyperplane parameters for each center.
        omega = centers
        gamma = -0.5 * np.linalg.norm(centers, axis=1) ** 2
        regions = []
        A_ini = initial_set.A
        b_ini = initial_set.b

        for j in range(K):
            A = np.empty((K - 1, dim))
            b = np.empty((K - 1, 1))
            row_idx = 0
            for h in range(K):
                if h != j:
                    A[row_idx] = omega[h] - omega[j]
                    b[row_idx] = gamma[j] - gamma[h]
                    row_idx += 1
            A = np.vstack((A, A_ini))
            b = np.vstack((b.reshape(-1, 1), b_ini.reshape(-1, 1)))
            regions.append({'A': A, 'b': b})
        return regions



    def plot_clusters_2d(self, samples, labels, centers=None, ax=None, alpha=0.7, s=20):
        """
        samples : (N,2) array
        labels  : (N,) cluster assignment
        centers : optional (K,2) array of centroids
        """

        samples = np.asarray(samples)
        labels = np.asarray(labels)

        if samples.shape[1] != 2:
            raise ValueError("samples must be 2D points (N,2)")

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        K = labels.max() + 1
        cmap = plt.cm.get_cmap("tab10", K)

        for k in range(K):
            pts = samples[labels == k]
            ax.scatter(pts[:, 0], pts[:, 1],
                       color=cmap(k),
                       s=s, alpha=alpha,
                       label=f"cluster {k}")

        if centers is not None:
            centers = np.asarray(centers)
            ax.scatter(centers[:, 0], centers[:, 1],
                       color="black",
                       marker="x",
                       s=120,
                       linewidths=3,
                       label="centers")

        ax.set_xlabel("x_1")
        ax.set_ylabel("x_2")
        ax.set_title("Clustered samples")
        # ax.legend()
        ax.axis("equal")
        ax.grid(True, alpha=0.2)

        plt.show()

    def bounding_box(self, samples: np.ndarray) -> np.ndarray:
        """
        Smallest box containing all samples.

        samples: array (N, n)
        returns: array (n, 2) with [lb, ub] per dimension
        """
        pts = np.asarray(samples, dtype=float)
        lb = pts.min(axis=0)
        ub = pts.max(axis=0)
        return np.column_stack((lb, ub))

    def split_bounds(self, bound: tuple, idxs: list) -> tuple:
        """
        Split a box (defined by lower and upper bounds) along the dimensions defined by idxs.

        Args:
            bound: Tuple (lb, ub), where lb and ub are 1D arrays representing lower and upper bounds.
            idxs: List of indices specifying which dimensions have to be split.

        Returns:
            A tuple of two new bounds (lb, new_bound) and (new_bound, ub) corresponding to the two halves.
        """
        lb, ub = bound
        idx_max_dist = (np.argmax(ub[idxs] - lb[idxs], keepdims=True)[0] + idxs[0])
        new_element = (lb[idx_max_dist] + ub[idx_max_dist]) / 2
        new_lb_1, new_ub_1 = lb.copy(), ub.copy()
        new_ub_1[idx_max_dist] = new_element
        new_lb_2, new_ub_2 = lb.copy(), ub.copy()
        new_lb_2[idx_max_dist] = new_element
        return (new_lb_1, new_ub_1), (new_lb_2, new_ub_2)

    def split_iteratively(self, lb: np.ndarray, ub: np.ndarray, K: int, idxs: list[int]) -> tuple:
        """
        Iteratively split a hyper-box along the given dimensions to create K sub-boxes.

        Args:
            lb: Lower bounds of the original box.
            ub: Upper bounds of the original box.
            K: Desired number of partitions.
            idxs: Indices of dimensions along which to split.

        Returns:
            polytopes: List of dictionaries {A, b} representing polyhedral constraints for each sub-box.
            boxes: List of K boxes, from derived polytopes but defined by lower and upper bounds.
        """
        bounds = [(lb, ub)]
        polytopes = []
        boxes = []
        for j in range(K - 1):
            max_dist = []
            for bound in bounds:
                lb_i, ub_i = bound
                max_dist.append(np.max(ub_i[idxs] - lb_i[idxs]))
            idx_bound_to_split = np.argmax(max_dist, keepdims=True)[0]
            new_bound_1, new_bound_2 = self.split_bounds(bounds[idx_bound_to_split], idxs)
            bounds[idx_bound_to_split] = new_bound_1
            bounds.append(new_bound_2)
        for j in range(K):
            lb_j = bounds[j][0]
            ub_j = bounds[j][1]
            uncertainty_box = np.array([lb_j, ub_j]).transpose()
            boxes.append(uncertainty_box)
            box = pt.box2poly(uncertainty_box)
            poly = {'A': box.A, 'b': box.b.reshape(-1, 1)}
            polytopes.append(poly)
        return polytopes, boxes

    def box_partitioning(self, K: int, lb: np.ndarray, ub: np.ndarray) -> tuple:
        """
        Partition a box (defined by lb and ub) into K sub-polytopes.
        Args:
            K (int): Number of partitions.
            lb (np.ndarray): Lower bound vector.
            ub (np.ndarray): Upper bound vector.
        Returns:
            list: A list of sub-polytopes, each represented as a dict with keys 'A' and 'b'.
            list: A list of np.arrays.
        """
        regions, boxes = self.split_iteratively(lb, ub, K, np.arange(len(lb)))
        return regions, boxes

    def split_gridding(self,  lb: np.ndarray, ub: np.ndarray, M: np.ndarray, split_idx: list, proportions=None):

        """
        Grid creating M_i intervals per coordinate i
        lb, ub : arrays shape (n,)
        split_idx : iterable of coordinates to split (length d)

        M : array-like length d
            M[i] = number of subintervals along dimension split_idx[i]

        proportions : None OR list/tuple of length d
            proportions[i] is array-like of length M[j] with positive entries.
            If provided, subinterval lengths along dim split_idx[j] follow these weights.

        returns (polytopes, out)
          - polytopes: list of dicts {'A': ..., 'b': ...}
          - boxes: list of arrays (n,2)
        """

        base = np.column_stack([lb, ub])

        d = len(split_idx)
        M = np.asarray(M, dtype=int).ravel()

        if d == 0:
            box_poly = pt.box2poly(base.copy())
            poly = {'A': box_poly.A, 'b': box_poly.b.reshape(-1, 1)}
            return [poly], [base.copy()]

        intervals_list = []
        for j, i in enumerate(split_idx):
            if proportions is None:
                grid = np.linspace(lb[i], ub[i], M[j] + 1)
            else:
                w = np.asarray(proportions[j], dtype=float)
                w = w / w.sum()
                grid = lb[i] + (ub[i] - lb[i]) * np.concatenate(([0.0], np.cumsum(w)))

            intervals = np.column_stack([grid[:-1], grid[1:]])
            intervals_list.append(intervals)

        boxes = []
        polytopes = []

        for choice in product(*intervals_list):
            box = base.copy()
            for j, i in enumerate(split_idx):
                box[i] = choice[j]
            boxes.append(box)

            box_poly = pt.box2poly(box)
            poly = {'A': box_poly.A, 'b': box_poly.b.reshape(-1, 1)}
            polytopes.append(poly)

        return polytopes, boxes


    def compute_partition_elements(
            self, samples: np.ndarray, regions: list, boxes: list, N_samples: int, traslate: bool = True
    ):
        """
        Compute relevant elements for each set.

        Returns:
            tuple: (p_hat, nominal_scen, polytopes, regions_new, clusters)
                - p_hat: probability of a sample being in the region.
                - nominal_scen: region center.
                - regions_new: regions with at least one sample (= non-zero probability), traslated to origing if traslate=True
                - boxes_new: regions with at least one sample (as boxes), traslated to origing if traslate=True
                - clusters: samples that belong to each region.
        """
        p_hat, nominal_scen, regions_new, boxes_new, clusters = [], [], [], [], []
        for j in range(len(regions)):
            region = regions[j]
            # Identify samples in the region.
            condition = np.all(samples @ region['A'].T <= region['b'].T, axis=1)
            scen = samples[condition]
            if scen.size:
                p_hat.append(len(scen) / N_samples)
                center = np.mean(scen, axis=0)
                nominal_scen.append(center)
                # Translate the polytope so the center is at the origin.
                b_translated = region['b'] - (region['A'] @ center).reshape(-1, 1) if traslate else region['b']
                regions_new.append({'A': region['A'], 'b': b_translated})
                if boxes is not None:
                    box_j_translated = boxes[j] - center.reshape(-1,1) if traslate else boxes[j]
                    boxes_new.append(box_j_translated)
                clusters.append(scen)
        return np.array(p_hat), np.array(nominal_scen), regions_new, boxes_new, clusters


class TruncatedGaussianNoise(BasicUncertainty):
    """
    Truncated noise model.

    Attributes:
        mean (np.ndarray): Mean of the corresponding Gaussian
        stdev (np.ndarray): St. dev. of the corresponding Gaussian
        interval (np.ndarray): Bounds of domain (boxes) (interval[i,0] contains lower limit for dimension-i,
            interval[i,1] contains upper limit for dimension-i)
    """

    def __init__(self, PARAMS: dict):
        super().__init__(PARAMS['n_uncertainty'])
        self.mean = PARAMS['mean_vec']
        self.stdev = PARAMS['std_vec']
        self.interval = PARAMS['intervals']

    def get_samples(self, Nsamples: int, horizon: int, seed: int=1, n_batches: int=1) -> np.ndarray:
        """
        Generate samples from a truncated gaussian distribution.

        Args:
            Nsamples (int): Number of samples.
            horizon (int): Time horizon.
        Returns:
            np.ndarray: Samples of shape (Nsamples, n_uncertainty * horizon).
        """
        mean_horizon = np.tile(self.mean, horizon)
        stdev_horizon = np.tile(self.stdev, horizon)
        low_horizon = np.tile(self.interval[:, 0], horizon)
        high_horizon = np.tile(self.interval[:, 1], horizon)
        return self.my_trunc_norm(mean_horizon, stdev_horizon, self.n_uncertainty * horizon, low_horizon, high_horizon,
                                  Nsamples, seed, n_batches)


    def get_domain(self, N_horizon: int) -> tuple:
        """
        Compute the domain for a truncated gaussian distribution.

        Args:
            N_horizon (int): Number of time steps.

        Returns:
            tuple: (lb, ub) arrays of length n_uncertainty * N_horizon.
        """
        lb = np.empty(self.n_uncertainty * N_horizon)
        ub = np.empty(self.n_uncertainty * N_horizon)

        for i in range(self.n_uncertainty):
            lb_val = self.interval[i][0]
            ub_val = self.interval[i][1]
            lb_i = np.full(N_horizon, lb_val)
            ub_i = np.full(N_horizon, ub_val)
            lb[i::self.n_uncertainty] = lb_i
            ub[i::self.n_uncertainty] = ub_i
        return lb, ub



class ParametricSinusoid(BasicUncertainty):
    """Parmetric sinusoid: y = a sin(omega k)"""

    def __init__(self, n_uncertainty:int, a_min:float, a_max:float, a_stdev:float, omega_min:float, omega_max:float,
                 omega_std:float, params_distribution: str) -> None:
        super().__init__(n_uncertainty)
        self.a_min = a_min
        self.a_max = a_max
        self.a_stdev = a_stdev
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.omega_std = omega_std
        self.params_distribution = params_distribution
        self.interval = np.arange(1, self.n_uncertainty+1)


    def sample_params(self, N: int, seed: int, n_batches: int=1) -> np.ndarray:
        """
        Sample parameters of sinusoid
        """
        rng = np.random.default_rng(seed)
        # a = rng.uniform(low=self.a_min, high=self.a_max, size=N)
        if self.params_distribution == 'normal':
            a_mean = (self.a_max + self.a_min) / 2.0
            a = self.my_trunc_norm(a_mean, self.a_stdev, 1, self.a_min, self.a_max, N, seed).squeeze()
            omega_mean = (self.omega_max + self.omega_min) / 2.0
            omega = self.my_trunc_norm(omega_mean, self.omega_std, 1, self.omega_min, self.omega_max, N, seed, n_batches).squeeze()
        elif self.params_distribution == 'uniform':
            a = rng.uniform(low=self.a_min, high=self.a_max, size=(n_batches, 1, N))
            omega = rng.uniform(low=self.omega_min, high=self.omega_max, size=(n_batches, 1, N))
        else:
            raise ValueError('Unknown distribution %s' % self.params_distribution)
        samples = []
        for i in range(n_batches):
            samples_i = np.vstack((a[i], omega[i]))
            samples.append(samples_i.T)
        return samples


    def from_params_to_sin(self, params: np.ndarray) -> np.ndarray:
        """
        Turn parameters samples into y_i = a_i sin(w_i k)
        """
        a = params[:, 0]
        omega = params[:, 1]
        interval = self.interval

        Y = a[:, None] * np.sin(
            omega[:, None] * interval[None, :]  # broadcast to (N,n)
        )
        return Y


    def get_samples(self, N: int, seed: int) -> np.ndarray:
        """
        Get samples y_i = a_i sin(w_i k)
        """
        n_batches = 1
        samples = self.sample_params(N, seed, n_batches)[0]

        Y = self.from_params_to_sin(samples)

        return Y

    def sinusoid_bounds(self, a_min: float, a_max: float,
                   w_min: float, w_max: float):
        """
        Returns bounds for y_k = a sin(w k)
        for k in S.

        Output: array shape (|S|, 2) -> [lower, upper]
        """

        S = self.interval

        out = np.zeros((len(S), 2), dtype=float)

        def sin_interval_extrema(theta_min, theta_max):
            """
            exact min/max of sin on interval: critical points can be stationary, or extrema
            """

            # candidates: extrema
            candidates = [np.sin(theta_min), np.sin(theta_max)]

            # add stationary points sin=1 (if any). Solve: theta = 2 pi n + pi/2, for theta in [theta_min, theta_max]
            n1_low = int(np.ceil((theta_min - np.pi / 2) / (2 * np.pi)))
            n1_high = int(np.floor((theta_max - np.pi / 2) / (2 * np.pi)))
            for n in range(n1_low, n1_high + 1):
                candidates.append(1.0)

            # add stationary points sin=-1 (if any). Solve: theta = 2 pi n + 3pi/2, for theta in [theta_min, theta_max]
            n2_low = int(np.ceil((theta_min - 3 * np.pi / 2) / (2 * np.pi)))
            n2_high = int(np.floor((theta_max - 3 * np.pi / 2) / (2 * np.pi)))
            for n in range(n2_low, n2_high + 1):
                candidates.append(-1.0)

            return min(candidates), max(candidates)

        for i, k in enumerate(S):
            theta_min = w_min * k
            theta_max = w_max * k

            # This finds max of sin(theta) for theta in [w_min K, w_max K]
            s_min, s_max = sin_interval_extrema(theta_min, theta_max)

            # combine with amplitude interval
            candidates = [
                a_min * s_min, a_min * s_max,
                a_max * s_min, a_max * s_max
            ]

            out[i, 0] = min(candidates)
            out[i, 1] = max(candidates)

        return out



    def grid_params(self, M_grid: np.ndarray):
        """
        Grid parameters space (A, W) defining y = a sin(wt)
        """
        domain = self.get_domain()
        lb = domain[:, 0]
        ub = domain[:, 1]

        n_params = lb.shape[0]

        split_idx = np.arange(n_params, dtype=int)
        params_partition = self.split_gridding(lb, ub, M=M_grid, split_idx=split_idx)

        return params_partition


    def get_domain(self):
        """
        Compute domain of parameter space
        """
        a_domain = np.array([self.a_min, self.a_max])
        omega_domain = np.array([self.omega_min, self.omega_max])

        return np.vstack((a_domain, omega_domain))

    def partition_from_params(self, boxes_params: list, nominal_scen: np.ndarray, traslate:bool=True):
        """
        From partition in parameters space, compute box uncertainty sets for uncertainty space
        """
        uncertainty_regions = []
        uncertainty_boxes = []

        for j in range(len(boxes_params)):
            box_params = boxes_params[j]
            center = nominal_scen[j]

            a_min, a_max = box_params[0]
            omega_min, omega_max = box_params[1]

            # Note: boxes_params are NOT traslated! I.e. just compute min and max over (non-traslated) intervals
            # for a and omega, of a*sin(omega k).
            uncertainty_box = self.sinusoid_bounds(a_min, a_max, omega_min, omega_max)

            # Now traslate (Note: nominal_scen refers to a^j * sin(omega^j k))
            uncertainty_box_traslated = uncertainty_box - center.reshape(-1, 1) if traslate else uncertainty_box

            uncertainty_boxes.append(uncertainty_box_traslated)
            poly = pt.box2poly(uncertainty_box_traslated)
            uncertainty_region = {'A': poly.A, 'b': poly.b.reshape(-1, 1)}
            uncertainty_regions.append(uncertainty_region)

        return uncertainty_regions, uncertainty_boxes