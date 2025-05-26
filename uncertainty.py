import numpy as np
import matplotlib.pyplot as plt
import polytope as pt
from sklearn.cluster import KMeans


class BasicUncertainty:
    """
    Base class for uncertainty handling.

    Args:
        n_uncertainty (int): Number of uncertainty dimensions.
        domain: Polytope representing the uncertainty domain.
    """

    def __init__(self, n_uncertainty: int):
        """
        Initialize class.
        """
        self.n_uncertainty = n_uncertainty
        self.domain = None

    def plot_samples(self, samples: np.ndarray, t_ini: float, tau: float) -> None:
        """
        Plot sample trajectories for each uncertainty component.

        Args:
            samples (np.ndarray): A list or array of sample vectors.
            t_ini (float): Initial time.
            tau (float): Time step size.
        """
        # Determine the horizon length based on the sample size and n_uncertainty.
        n_horizon = int(samples[0].size / self.n_uncertainty)
        t_interval = np.linspace(t_ini, t_ini + int(n_horizon * tau), n_horizon)

        # Create n_uncertainty subplots (vertically arranged)
        fig, axes = plt.subplots(self.n_uncertainty, 1, figsize=(8, 6))
        if self.n_uncertainty == 1:
            axes = [axes]

        # Loop over each sample and uncertainty dimension.
        for sample in samples:
            for i in range(self.n_uncertainty):
                # Extract every n_uncertainty-th element starting at index i.
                axes[i].plot(t_interval, sample[i::self.n_uncertainty])
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel(f"Output {i + 1}")
                axes[i].grid(True)

        plt.tight_layout()
        plt.show()

    def set_domain(self, lb: np.ndarray, ub: np.ndarray) -> None:
        """
        Set the uncertainty domain (a box) as a polytope.

        Args:
            lb (np.ndarray): Lower bounds.
            ub (np.ndarray): Upper bounds.
        """
        self.domain = pt.Polytope(lb, ub)

    def my_kmeans_partition(self, samples: np.ndarray, K: int, initial_set: pt.Polytope, dim: int) -> list:
        """
        Partition the sample space using KMeans clustering and a Voronoi partitioning.

        Args:
            samples (np.ndarray): Sample data.
            K (int): Number of clusters.
            initial_set: uncertainty domain.
            dim (int): Dimensionality of the space.

        Returns:
            list: A list of regions defined by hyperplanes.
        """
        clustering_algo = KMeans(n_clusters=K, init='k-means++')
        clustering_algo.fit(samples)
        centers = clustering_algo.cluster_centers_
        regions = self.my_voronoi(K, centers, initial_set, dim)
        return regions

    def my_voronoi(self, K: int, centers: np.ndarray, initial_set: pt.Polytope, dim: int) -> list:
        """
        Construct a Voronoi partition of the space using hyperplanes.

        Args:
            K (int): Number of clusters.
            centers (np.ndarray): Cluster centers.
            initial_set: uncertainty domain.
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

    def box_partitioning(self, K: int, lb: np.ndarray, ub: np.ndarray) -> list:
        """
        Partition a box (defined by lb and ub) into K boxes.
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


    def compute_partition_elements(
            self, samples: np.ndarray, regions: list, boxes: list, N_samples: int
    ):
        """
        Compute relevant elements for each set.

        Returns:
            tuple: (p_hat, nominal_scen, polytopes, regions_new, clusters)
                - p_hat: probability of a sample being in the region.
                - nominal_scen: region center.
                - polytopes: translated polytopes (to origin)
                - regions_new: regions with at least one sample (= non-zero probability).
                - boxes_new: regions with at least one sample (as boxes).
                - clusters: samples that belong to each region.
        """
        p_hat, nominal_scen, polytopes, regions_new, boxes_new, clusters = [], [], [], [], [], []
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
                b_translated = region['b'] - (region['A'] @ center).reshape(-1, 1)
                polytopes.append({'A': region['A'], 'b': b_translated})
                regions_new.append(region)
                if boxes is not None:
                    boxes_new.append(boxes[j])
                clusters.append(scen)
        return np.array(p_hat), np.array(nominal_scen), polytopes, regions_new, boxes_new, clusters


class SinusoidalDisturbance(BasicUncertainty):
    """
    Sinusoidal uncertainty model:
        y_k = a_k * sin(omega * k + p) + w_k
    where:
        a_k = Uniform([a_min * k, a_max * k])
        omega = Gaussian mixture (omega_means, omega_std, omega_weights)
        p = Uniform([p_min, p_max])
        w_k = Uniform([w_min * (k+1), w_max * (k+1)])
        for k = 0,...,N-1.
    """

    def __init__(self, PARAMS: dict):
        """
        Initialize sinusodial uncertainty model with given parameters.
        """
        super().__init__(PARAMS['n_uncertainty'])
        self.a_min = PARAMS['a_min']
        self.a_max = PARAMS['a_max']
        self.omega_means = PARAMS['omega_means']
        self.omega_std = PARAMS['omega_std']
        self.omega_weights = PARAMS['omega_weights']
        self.p_min = PARAMS['p_min']
        self.p_max = PARAMS['p_max']
        self.w_min = PARAMS['w_min']
        self.w_max = PARAMS['w_max']

    def get_samples(self, Nsamples: int, horizon: int, t_ini: float = 0) -> np.ndarray:
        """
        Generate sinusoidal samples.

        Args:
            Nsamples (int): Number of samples.
            horizon (int): Time horizon length.
            t_ini (float): Initial time.

        Returns:
            np.ndarray: Samples of shape (Nsamples, n_uncertainty * horizon).
        """
        k_values = np.arange(t_ini, t_ini + horizon)
        Y = np.empty((Nsamples, self.n_uncertainty * horizon))

        for i in range(self.n_uncertainty):
            # Draw omega from a Gaussian mixture for dimension i.
            omega = (np.random.choice(
                a=self.omega_means[i],
                p=self.omega_weights[i],
                size=(Nsamples, 1))
                + np.random.normal(0, self.omega_std[i], (Nsamples, 1)))
            p = np.random.uniform(self.p_min[i], self.p_max[i], (Nsamples, 1))
            a_k = np.random.uniform(
                self.a_min[i] * (k_values - t_ini + 1),
                self.a_max[i] * (k_values - t_ini + 1),
                (Nsamples, horizon))
            w_k = np.random.uniform(
                self.w_min[i] * (k_values - t_ini + 1),
                self.w_max[i] * (k_values - t_ini + 1),
                (Nsamples, horizon))
            y_k = a_k * np.sin(omega * k_values + p) + w_k
            Y[:, i::self.n_uncertainty] = y_k
        return Y

    def get_domain(self, N_horizon: int) -> tuple:
        """
        Compute the domain for the sinusoidal disturbance over a time horizon.

        Args:
            N_horizon (int): Horizon length.

        Returns:
            tuple: (lb, ub) where each is an array of length n_uncertainty * N_horizon.
        """
        lb = np.empty(self.n_uncertainty * N_horizon)
        ub = np.empty(self.n_uncertainty * N_horizon)

        for i in range(self.n_uncertainty):
            lb_i = (-self.a_max[i] + self.w_min[i]) * np.arange(1, N_horizon + 1)
            ub_i = (self.a_max[i] + self.w_max[i]) * np.arange(1, N_horizon + 1)
            lb[i::self.n_uncertainty] = lb_i
            ub[i::self.n_uncertainty] = ub_i
        return lb, ub

