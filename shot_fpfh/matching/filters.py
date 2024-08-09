"""
Filter functions that can be used to filter matches between descriptors based on the distances between each pair of
matched descriptors.
"""

import numpy as np
import numpy.typing as npt


def threshold_filter(
    distances: npt.NDArray[np.float64],
    threshold_multiplier: float,
) -> np.ndarray[bool]:
    return distances <= distances[distances.nonzero()[0]].min() * threshold_multiplier


def quantile_filter(
    distances: npt.NDArray[np.float64],
    quantiles: tuple[float, float],
) -> np.ndarray[bool]:
    threshold_values = np.quantile(distances, quantiles)
    return (distances >= threshold_values[0]) & (distances <= threshold_values[1])


def left_median_filter(
    distances: npt.NDArray[np.float64],
) -> np.ndarray[bool]:
    median_dist = np.median(distances)
    return (distances <= median_dist) & (
        distances >= (median_dist + distances.nonzero()[0].min()) / 2
    )
