import numpy as np

from scipy.interpolate import interpolate

from .IM import IM, IMType
from .HazardResult import HazardResult


def get_min_max_im_level(im: IM):
    """Get minimum and maximum for the given im. Values for velocity are
    given on cm/s, acceleration on cm/s^2 and Ds on s
    """
    if im.is_pSA:
        assert im.period is not None, "No period provided for pSA, this is an error"
        if im.period <= 0.5:
            return 0.005, 10.0
        elif 0.5 < im.period <= 1.0:
            return 0.005, 7.5
        elif 1.0 < im.period <= 3.0:
            return 0.0005, 5.0
        elif 3.0 < im.period <= 5.0:
            return 0.0005, 4.0
        elif 5.0 < im.period <= 10.0:
            return 0.0005, 3.0
    if im.im_type is IMType.PGA:
        return 0.0001, 10.0
    elif im.im_type is IMType.PGV:
        return 1.0, 400.0
    elif im.im_type is IMType.CAV:
        return 0.0001 * 980, 20.0 * 980.0
    elif im.im_type is IMType.AI:
        return 0.01, 1000.0
    elif im.im_type is IMType.Ds575 or im.im_type is IMType.Ds595:
        return 1.0, 400.0
    else:
        raise ValueError(
            "Unknown IM, cannot generate a range of IM values. Exiting the program"
        )


def get_im_levels(im: IM, n_values: int = 100):
    """
    Create a range of values for a given IM according
    to their min, max as defined by get_min_max_values

    Parameters
    ----------
    im: IM
        The IM Object to get im values for
    n_values: int

    Returns
    -------
    Array of IM values
    """
    start, end = get_min_max_im_level(im)
    im_values = np.logspace(
        start=np.log(start), stop=np.log(end), num=n_values, base=np.e
    )
    return im_values


def exceedance_to_im(exceedance: float, hazard_result: HazardResult):
    """
    Converts the given exceedance rate to an IM value

    Parameters
    ----------
    exceedance: float
    hazard_result: HazardResult

    Returns
    -------
    float
        The IM value corresponding to the given exceedance
    """
    return np.exp(
        interpolate.interp1d(
            np.log(hazard_result.mean_hazard.values) * -1,
            np.log(hazard_result.im_levels),
            kind="linear",
            bounds_error=True,
        )(np.log(exceedance) * -1)
    )


def im_to_exceedance(im_value: float, hazard_result: HazardResult):
    """
    Convert from IM value to exceedance rate

    Parameters
    ----------
    im_value: float
    hazard_result: HazardResult

    Returns
    -------
    float
    """
    return np.exp(
        interpolate.interp1d(
            np.log(hazard_result.im_levels),
            np.log(hazard_result.mean_hazard.values),
            kind="linear",
            bounds_error=True,
        )(np.log(im_value))
    )