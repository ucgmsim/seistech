import numpy as np


def rp_to_prob(rp: float, t: float = 1.0):
    """
    Converts return period to exceedance probability
    Based on Poisson distribution

    Parameters
    ----------
    rp: float
        Return period
    t: float
        Time period of interest

    Returns
    -------
    Exceedance probability
    """
    return 1 - np.exp(-t / rp)
