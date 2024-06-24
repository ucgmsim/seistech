"""Implementation of the Loth & Baker (2013) site correlation model"""
import numpy as np

valid_periods = np.array([0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0])
short_range_corregionalization = np.array(
    [
        [0.30, 0.24, 0.23, 0.22, 0.16, 0.07, 0.03, 0.00, 0.00],
        [0.24, 0.27, 0.19, 0.13, 0.08, 0.00, 0.00, 0.00, 0.00],
        [0.23, 0.19, 0.26, 0.19, 0.12, 0.04, 0.00, 0.00, 0.00],
        [0.22, 0.13, 0.19, 0.32, 0.23, 0.14, 0.09, 0.06, 0.04],
        [0.16, 0.08, 0.12, 0.23, 0.32, 0.22, 0.13, 0.09, 0.07],
        [0.07, 0.00, 0.04, 0.14, 0.22, 0.33, 0.23, 0.19, 0.16],
        [0.03, 0.00, 0.00, 0.09, 0.13, 0.23, 0.34, 0.29, 0.24],
        [0.00, 0.00, 0.00, 0.06, 0.09, 0.19, 0.29, 0.30, 0.25],
        [0.00, 0.00, 0.00, 0.04, 0.07, 0.16, 0.24, 0.25, 0.24],
    ]
)

long_range_corregionalization = np.array(
    [
        [0.31, 0.26, 0.27, 0.24, 0.17, 0.11, 0.08, 0.06, 0.05],
        [0.26, 0.29, 0.22, 0.15, 0.07, 0.00, 0.00, 0.00, -0.03],
        [0.27, 0.22, 0.29, 0.24, 0.15, 0.09, 0.03, 0.02, 0.00],
        [0.24, 0.15, 0.24, 0.33, 0.27, 0.23, 0.17, 0.14, 0.14],
        [0.17, 0.07, 0.15, 0.27, 0.38, 0.34, 0.23, 0.19, 0.21],
        [0.11, 0.00, 0.09, 0.23, 0.34, 0.44, 0.33, 0.29, 0.32],
        [0.08, 0.00, 0.03, 0.17, 0.23, 0.33, 0.45, 0.42, 0.42],
        [0.06, 0.00, 0.02, 0.14, 0.19, 0.29, 0.42, 0.47, 0.47],
        [0.05, -0.03, 0.00, 0.14, 0.21, 0.32, 0.42, 0.47, 0.54],
    ]
)

nugget_corregionalization = np.array(
    [
        [0.38, 0.36, 0.35, 0.17, 0.04, 0.04, 0.00, 0.03, 0.08],
        [0.36, 0.43, 0.35, 0.13, 0.00, 0.02, 0.00, 0.02, 0.08],
        [0.35, 0.35, 0.45, 0.11, -0.04, -0.02, -0.04, -0.02, 0.03],
        [0.17, 0.13, 0.11, 0.35, 0.20, 0.06, 0.02, 0.04, 0.02],
        [0.04, 0.00, -0.04, 0.20, 0.30, 0.14, 0.09, 0.12, 0.04],
        [0.04, 0.02, -0.02, 0.06, 0.14, 0.22, 0.12, 0.13, 0.09],
        [0.00, 0.00, -0.04, 0.02, 0.09, 0.12, 0.21, 0.17, 0.13],
        [0.03, 0.02, -0.02, 0.04, 0.12, 0.13, 0.17, 0.23, 0.10],
        [0.08, 0.08, 0.03, 0.02, 0.04, 0.09, 0.13, 0.10, 0.22],
    ]
)


def get_correlations(im_1: str, im_2: str, site_dist: np.ndarray) -> np.ndarray:
    """
    Computes the spatial cross-correlation
    as per the Loth & Baker model

    Note: This implementation doesn't currently support
        interpolation between the defined periods.
        Instead, it just snaps to the lower period.

    Parameters
    ----------
    im_1: str
    im_2: str
    site_dist: array of floats
        The site-to-site distances

    Returns
    -------
    cov: array of floats
        The cross-correlation
    """
    T_1 = float(im_1.split("_")[1])
    T_2 = float(im_2.split("_")[1])

    idx_1 = np.argmin(T_1 >= valid_periods)
    idx_2 = np.argmin(T_2 >= valid_periods)
    cov = short_range_corregionalization[idx_1, idx_2] * np.exp(
        -3.0 * site_dist / 20.0
    ) + long_range_corregionalization[idx_1, idx_2] * np.exp(-3.0 * site_dist / 70.0)

    self_mask = np.isclose(site_dist, 0.0)
    cov[self_mask] += nugget_corregionalization[idx_1, idx_2]
    return cov
