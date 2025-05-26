import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, Tuple, List

def compute_stats(arr: np.ndarray, experiments: List[Any], axis: int = 1) -> pd.DataFrame:
    """
    Compute min, max, average, and standard deviation, empirically over experiments.
    Returns a DataFrame with statistics.
    """
    stats = {
        'min': np.min(arr, axis=axis),
        'max': np.max(arr, axis=axis),
        'avg': np.mean(arr, axis=axis),
        'std': np.std(arr, axis=axis)
    }
    # Create a DataFrame where each row is a statistic and each column is an experiment.
    return pd.DataFrame(stats, index=experiments).T


def check_probability_sum(p_hat: np.ndarray, tol: float = 1e-6) -> None:
    """
    Checks if probabilities sum to one. Raise a warning if they do not.
    """
    total = np.sum(p_hat)
    if abs(total - 1) > tol:
        warnings.warn(f"Probabilities do not sum to 1.")


def filter_feasible_experiments(store_violations: np.ndarray, arrays: List[np.ndarray]) -> List[np.ndarray]:
    """
    Filters out experiments where violations are -1 (infeasible experiments).
    Returns a list of filtered arrays.
    """
    mask = (store_violations >= 0).all(axis=0)
    return [arr[:, mask] for arr in arrays]