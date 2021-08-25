import pandas as pd
import numpy as np

import seistech_calc.utils as utils
import seistech_calc.constants as const
from qcore.nhm import load_nhm

RUPTURE_FAULT_DF_COLUMNS = [
    "rupture_name",
    "annual_rec_prob",
    "magnitude",
    "tectonic_type",
]
RUPTURE_DS_DF_COLUMNS = ["rupture_name", "annual_rec_prob", "magnitude"]


def rupture_df_from_erf(
    erf_ffp: str, erf_file_type: const.ERFFileType = const.ERFFileType.flt_nhm
):
    """Creates a standardised pandas dataframe for the
    ruptures in the given erf file.

    Parameters
    ----------
    erf_ffp : str
        Path to the ERF file
    erf_file_type : ERFFileType
        Type of the erf file

    Returns
    -------
    DataFrame
        index = rupture id
        columns = [rupture name, annual recurrence probability, magnitude, tectonic type]
    """
    erf_name = utils.get_erf_name(erf_ffp)
    if erf_file_type is const.ERFFileType.flt_nhm:
        nhm_infos = load_nhm(erf_ffp)

        rupture_dict = {
            f"{info.name}_{erf_name}": [
                info.name,
                1 / info.recur_int_median,
                info.mw,
                info.tectonic_type,
            ]
            for key, info in nhm_infos.items()
            if info.recur_int_median > 0
        }

        return pd.DataFrame.from_dict(
            rupture_dict, orient="index", columns=RUPTURE_FAULT_DF_COLUMNS
        ).sort_index()

    elif erf_file_type is const.ERFFileType.ds_erf:
        df = pd.read_csv(erf_ffp)
        df.columns = RUPTURE_DS_DF_COLUMNS
        df.index = rupture_name_to_id(df.rupture_name.values.astype(str), erf_ffp)
        return df.sort_index()

    raise NotImplementedError("The specified erf file type is currently not supported")


def rupture_name_to_id(rupture_names: np.ndarray, erf_ffp: str):
    """Converts the given ruptures names to rupture ids

    Parameters
    ----------
    rupture_names: numpy array of strings
    erf_name: str

    Returns
    -------
    numpy array of strings
    """
    return np.char.add(rupture_names.astype(str), f"_{utils.get_erf_name(erf_ffp)}")
