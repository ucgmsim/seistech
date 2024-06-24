import os
import math
import warnings

import numpy as np
import pandas as pd

import gmhazard_calc.constants as const
from gmhazard_calc.im import IM, IMType
from qcore import nhm


def calculate_rupture_rates(
    nhm_df: pd.DataFrame,
    rup_name: str = "rupture_name",
    annual_rec_prob_name: str = "annual_rec_prob",
    mag_name: str = "mag_name",
) -> pd.DataFrame:
    """Takes in a list of background ruptures and
    calculates the rupture rates for the given magnitudes

    The rupture rate calculation is based on the Gutenberg-Richter equation from OpenSHA.
    It discretises the recurrance rate per magnitude instead of storing the probability of
    rupture exceeding a certain magnitude
    https://en.wikipedia.org/wiki/Gutenberg%E2%80%93Richter_law
    https://github.com/opensha/opensha-core/blob/master/src/org/opensha/sha/magdist/GutenbergRichterMagFreqDist.java

    Also includes the rupture magnitudes
    """
    data = np.ndarray(
        sum(nhm_df.n_mags),
        dtype=[
            (rup_name, str, 64),
            (annual_rec_prob_name, np.float64),
            (mag_name, np.float64),
        ],
    )

    # Make an array of fault bounds so the ith faults has
    # the ruptures indexes[i]-indexes[i+1]-1 (inclusive)
    indexes = np.cumsum(nhm_df.n_mags.values)
    indexes = np.insert(indexes, 0, 0)

    index_mask = np.zeros(len(data), dtype=bool)

    warnings.filterwarnings(
        "ignore", message="invalid value encountered in true_divide"
    )
    for i, line in nhm_df.iterrows():
        index_mask[indexes[i] : indexes[i + 1]] = True

        # Generate the magnitudes for each rupture
        sample_mags = np.linspace(line.M_min, line.M_cutoff, line.n_mags)

        for ii, iii in enumerate(range(indexes[i], indexes[i + 1])):
            data[rup_name][iii] = create_ds_rupture_name(
                line.source_lat,
                line.source_lon,
                line.source_depth,
                sample_mags[ii],
                line.tect_type,
            )

        # Calculate the cumulative rupture rate for each rupture
        baseline = (
            line.b
            * math.log(10, 2.72)
            / (1 - 10 ** (-1 * line.b * (line.M_cutoff - line.M_min)))
        )
        f_m_mag = np.power(10, (-1 * line.b * (sample_mags - line.M_min))) * baseline
        f_m_mag = np.append(f_m_mag, 0)
        rup_prob = (f_m_mag[:-1] + f_m_mag[1:]) / 2 * 0.1
        total_cumulative_rate = rup_prob * line.totCumRate

        # normalise
        total_cumulative_rate = (
            line.totCumRate * total_cumulative_rate / np.sum(total_cumulative_rate)
        )

        data[mag_name][index_mask] = sample_mags
        data[annual_rec_prob_name][index_mask] = total_cumulative_rate

        index_mask[indexes[i] : indexes[i + 1]] = False
    background_values = pd.DataFrame(data=data)
    background_values.fillna(0, inplace=True)

    return background_values


def convert_im_type(im_type: str):
    """Converts the IM type to the standard format,
    will be redundant in the future"""
    if im_type.startswith("SA"):
        return "p" + im_type.replace("p", ".")
    return im_type


def get_erf_name(erf_ffp: str) -> str:
    """Gets the erf name, required for rupture ids

    Use this function for consistency, instead of doing it manual
    """
    return os.path.basename(erf_ffp).split(".")[0]


def pandas_isin(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """This is the same as a np.isin,
    however is significantly faster for large arrays

    https://stackoverflow.com/questions/15939748/check-if-each-element-in-a-numpy-array-is-in-another-array
    """
    return pd.Index(pd.unique(array_2)).get_indexer(array_1) >= 0


def get_min_max_values_for_im(im: IM):
    """
    Get minimum and maximum for the given im. Values for velocity are
    given on cm/s, acceleration on cm/s^2 and Ds on s
    """
    if im.is_pSA():
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
        print("Unknown IM, cannot generate a range of IM values. Exiting the program")
        exit(1)


def get_im_values(im: IM, n_values: int = 100):
    """
    Create an range of values for a given IM according to their min, max
    as defined by get_min_max_values

    Parameters
    ----------
    im: IM
        The IM Object to get im values for
    n_values: int

    Returns
    -------
    Array of IM values
    """
    start, end = get_min_max_values_for_im(im)
    im_values = np.logspace(
        start=np.log(start), stop=np.log(end), num=n_values, base=np.e
    )
    return im_values


def closest_location(locations, lat, lon):
    """
    Find position of closest location in locations 2D np.array of (lat, lon).
    """
    d = (
        np.sin(np.radians(locations[:, 0] - lat) / 2.0) ** 2
        + np.cos(np.radians(lat))
        * np.cos(np.radians(locations[:, 0]))
        * np.sin(np.radians(locations[:, 1] - lon) / 2.0) ** 2
    )
    return np.argmin(6378.139 * 2.0 * np.arctan2(np.sqrt(d), np.sqrt(1 - d)))


def read_emp_file(emp_file, cs_faults):
    """Read an empiricial file"""

    # Read file
    emp = pd.read_csv(
        emp_file,
        sep="\t",
        names=("fault", "mag", "rrup", "med", "dev", "prob"),
        usecols=(0, 1, 2, 5, 6, 7),
        dtype={
            "fault": object,
            "mag": np.float32,
            "rrup": np.float32,
            "med": np.float32,
            "dev": np.float32,
            "prob": np.float32,
        },
        engine="c",
        skiprows=1,
    )

    # Type contains 0: Type A; 1: Type B; 2: Distributed Seismicity
    emp["type"] = pd.Series(0, index=emp.index, dtype=np.uint8)
    # Type B faults
    emp.type += np.invert(np.vectorize(cs_faults.__contains__)(emp.fault))
    # Distributed seismicity
    emp.loc[emp.fault == "PointEqkSource", "type"] = 2

    # Assume magnitude correct where prob is given
    mag, rrup = {}, {}

    # Get the unique faults, and their indices (first occurrence)
    unq_faults, unq_fault_ind = np.unique(emp.fault, return_index=True)

    # Sort by first occurrence
    sort_id = np.argsort(unq_fault_ind)
    unq_faults, unq_fault_ind = unq_faults[sort_id], unq_fault_ind[sort_id]

    # All faults except for distributed seismicity have incorrect probabilities (??)
    for i in range(unq_fault_ind.size - 1):
        cur_fault_rows = emp.iloc[unq_fault_ind[i] : unq_fault_ind[i + 1]]
        prob = cur_fault_rows.prob.values

        # Prevent new input rules being undetected
        assert np.sum(prob != 0) == 1

        # Fault properties
        mag[unq_faults[i]] = cur_fault_rows.mag[np.argmax(prob) + unq_fault_ind[i]]
        rrup[unq_faults[i]] = cur_fault_rows.rrup[unq_fault_ind[i]]

        # Because pandas is incapable of storing a view
        emp.iloc[unq_fault_ind[i] : unq_fault_ind[i + 1], 5] = np.average(prob)

    return emp, mag, rrup


def check_names(needles, haystack):
    """Check that the required elements (needles) exist in the header (haystack)
     and that there are the correct number of elements

    :param needles: List of elements to be checked
    :param haystack: List whose contents are to be checked
    :return: True if all elements are contained
    """
    n_expected_variables = len(needles)
    if len(haystack) == n_expected_variables and np.all(np.isin(needles, haystack)):
        return True
    else:
        raise ValueError(
            f"Elements are not as expected. Must have {n_expected_variables} in "
            f"the haystack ({', '.join(needles)})"
        )


def create_ds_rupture_name(lat, lon, depth, mag, tect_type):
    """
    A rupture name is unique for each distributed seismicity source. Each source has a unique empirical IM value and
    rate of exceedance. A fault is a common name for multiple ruptures at the same point (lat, lon, depth)

    :return: a string containing a unique rupture name for each distributed seismicity source
    """
    return "{}--{}_{}".format(create_ds_fault_name(lat, lon, depth), mag, tect_type)


def create_ds_fault_name(lat, lon, depth):
    """
    :return: a string containing a fault name for every rupture at that point (lat, lon, depth)
    """
    return "{}_{}_{}".format(lat, lon, depth)


def read_ds_nhm(background_file):
    """
    Reads a background seismicity file and returns a datafram with the columns of the file

    The txt file is formatted for OpenSHA
    """
    return pd.read_csv(
        background_file,
        skiprows=5,
        delim_whitespace=True,
        header=None,
        names=[
            "a",
            "b",
            "M_min",
            "M_cutoff",
            "n_mags",
            "totCumRate",
            "source_lat",
            "source_lon",
            "source_depth",
            "rake",
            "dip",
            "tect_type",
        ],
    )


def ds_nhm_to_rup_df(background_ffp):
    """
    :param background_ffp: nhm background txt filepath
    :return: a rupture df which contains for each unique rupture:
    rupture_name, fault_name, mag, dip, rake, dbot, dtop, tect_type and reccurance_rate

    This function does not calculate the recurrance rate and returns an empty column for that dtype
    """
    background_df = read_ds_nhm(background_ffp)
    data = np.ndarray(
        sum(background_df.n_mags),
        dtype=[
            ("rupture_name", str, 64),
            ("fault_name", str, 64),
            ("mag", np.float64),
            ("dip", np.float64),
            ("rake", np.float64),
            ("dbot", np.float64),
            ("dtop", np.float64),
            ("tect_type", str, 64),
            ("recurrance_rate", np.float64),
        ],
    )

    indexes = np.cumsum(background_df.n_mags.values)
    indexes = np.insert(indexes, 0, 0)
    index_mask = np.zeros(len(data), dtype=bool)

    for i, line in background_df.iterrows():
        index_mask[indexes[i] : indexes[i + 1]] = True

        # Generate the magnitudes for each rupture
        sample_mags = np.linspace(line.M_min, line.M_cutoff, line.n_mags)

        for ii, iii in enumerate(range(indexes[i], indexes[i + 1])):
            data["rupture_name"][iii] = create_ds_rupture_name(
                line.source_lat,
                line.source_lon,
                line.source_depth,
                sample_mags[ii],
                line.tect_type,
            )

        data["fault_name"][index_mask] = create_ds_fault_name(
            line.source_lat, line.source_lon, line.source_depth
        )
        data["rake"][index_mask] = line.rake
        data["dip"][index_mask] = line.dip
        data["dbot"][index_mask] = line.source_depth
        data["dtop"][index_mask] = line.source_depth
        data["tect_type"][index_mask] = line.tect_type
        data["mag"][index_mask] = sample_mags

        index_mask[indexes[i] : indexes[i + 1]] = False  # reset the index mask

    df = pd.DataFrame(data=data)
    df["fault_name"] = df["fault_name"].astype("category")
    df["rupture_name"] = df["rupture_name"].astype("category")
    df["tect_type"] = df["tect_type"].astype("category")

    return df


def flt_nhm_to_rup_df(nhm_ffp):
    """
    :param nhm_ffp: nhm fault txt filepath
    :return: a rupture df which contains for each unique rupture:
    rupture_name, fault_name, mag, dip, rake, dbot, dtop, tect_type and reccurance_rate
    """
    nhm_infos = nhm.load_nhm(nhm_ffp)

    rupture_dict = {
        i: [
            info.name,
            info.name,
            info.mw,
            info.dip,
            info.rake,
            info.dbottom,
            info.dtop,
            info.tectonic_type,
            info.recur_int_median,
        ]
        for i, (key, info) in enumerate(nhm_infos.items())
    }

    df = pd.DataFrame.from_dict(
        rupture_dict,
        orient="index",
        columns=[
            "fault_name",
            "rupture_name",
            "mag",
            "dip",
            "rake",
            "dbot",
            "dtop",
            "tect_type",
            "recurrance_rate",
        ],
    ).sort_index()

    df["fault_name"] = df["fault_name"].astype("category")
    df["rupture_name"] = df["rupture_name"].astype("category")
    df["tect_type"] = df["tect_type"].astype("category")

    return df


def create_parametric_db_name(
    model_name: str, source_type: const.SourceType, suffix: str = None
):
    """Returns the name of a parametric IMDB given a model and source type with an optional suffix"""
    suffix = f"_{suffix}" if suffix else ""
    return f"{model_name}_{source_type.value}{suffix}.db"


def to_mu_sigma(df: pd.DataFrame, im: IM):
    return df.loc[:, [str(im), f"{im}_sigma"]].rename(
        columns={str(im): "mu", f"{im}_sigma": "sigma"}
    )
