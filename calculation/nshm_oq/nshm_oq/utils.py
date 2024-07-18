import json
import re
import shutil
from pathlib import Path

import geojson
import numpy as np
import pandas as pd
from turfpy import measurement

import gmhazard_common as common
from . import constants


def get_backarc_mask(backarc_json_ffp: Path, locs: np.ndarray):
    """
    Computes a mask identifying each location
    that requires the backarc flag based on
    wether it is inside the backarc polygon or not

    locs: array of floats
        [lon, lat]

    """

    # Determine if backarc needs to be enabled for each loc
    points = geojson.FeatureCollection(
        [
            geojson.Feature(geometry=geojson.Point(tuple(cur_loc[::-1]), id=ix))
            for ix, cur_loc in enumerate(locs)
        ]
    )
    with backarc_json_ffp.open("r") as f:
        poly_coords = np.flip(json.load(f)["geometry"]["coordinates"][0], axis=1)

    polygon = geojson.Polygon([poly_coords.tolist()])
    backarc_ind = (
        [
            cur_point["geometry"]["id"]
            for cur_point in measurement.points_within_polygon(points, polygon)[
                "features"
            ]
        ],
    )
    backarc_mask = np.zeros(shape=locs.shape[0], dtype=bool)
    backarc_mask[backarc_ind] = True

    return backarc_mask


def calculate_z1pt0(vs30: float):
    """
    Calculate the depth to the 1.0 km/s shear wave velocity based on vs30 values.

    Parameters
    ----------
    vs30 : float
        The shear wave velocity (in m/s) at a depth of 30m.

    Returns
    -------
    float
        The depth to the 1.0 km/s velocity horizon (in m).

    References
    ----------
    Chiou, B., & Youngs, R. (2014). Update of the Chiou and Youngs NGA model
    for the average horizontal component of peak ground motion and response spectra.
    Earthquake Spectra, 30(3), 1117-1153.
    """
    c1 = 571**4.0
    c2 = 1360.0**4.0
    return np.exp((-7.15 / 4.0) * np.log((vs30**4.0 + c1) / (c2 + c1)))


def calculate_z2pt5_ngaw2(vs30: float):
    """
    Calculate the depth to the 2.5 km/s shear wave velocity
    based on vs30 values.

    Parameters
    ----------
    vs30 : float
        The shear wave velocity (in m/s) at a depth of 30 m.

    Returns
    -------
    float
        The depth to the 2.5 km/s velocity horizon (in km).

    References
    ----------
    Campbell, K.W., & Bozorgnia, Y. (2014). NGA-West2 ground motion model for the
    average horizontal components of PGA, PGV, and 5% damped linear acceleration response spectra.
    Earthquake Spectra, 30(3), 1087-1114.
    """
    c1 = 7.089
    c2 = -1.144
    z2pt5 = np.exp(c1 + np.log(vs30) * c2)
    return z2pt5


def setup_run_dir(
    output_dir: Path,
    source_def_dir: Path,
    site_df: pd.DataFrame,
):
    """
    Runs the overlapping setup steps for
    hazard and disagg

    Parameters
    ----------
    output_dir: Path
        Output directory for the hazard results
    source_def_dir: Path
        Source definition directory
    site_df: pd.DataFrame
        DataFrame containing the site information
        Required columns: lon, lat, vs30, z1p0, z2p5, backarc

    Returns
    -------
    run_dir: Path
        The run directory
    source_model_ffp: Path
        The source model file path
    """
    # Create the run dir
    (run_dir := output_dir / "run_dir").mkdir(exist_ok=False, parents=False)

    # Link the source definitions directory
    (local_source_def_dir := run_dir / "source_definitions").symlink_to(source_def_dir)

    # Copy the source model
    source_model_ffp = run_dir / "source_model.xml"
    shutil.copy(
        constants.RESOURCE_DIR / constants.BASE_SOURCE_MODEL_FNAME, source_model_ffp
    )

    # Update the source model xml file
    with source_model_ffp.open("r+") as f:
        base_source_model = f.read(-1)
        base_source_model = base_source_model.replace(
            "$SOURCE_DIR$", local_source_def_dir.name
        )
        f.seek(0)
        f.write(base_source_model)

    # Copy the GMM logic tree
    shutil.copy(
        constants.RESOURCE_DIR / constants.GMM_LOGIC_TREE_FNAME,
        run_dir / constants.GMM_LOGIC_TREE_FNAME,
    )

    # Write the site file
    site_df.to_csv(run_dir / "sites.csv", index=False)

    return run_dir, source_model_ffp


def to_std_im(oq_im: str):
    """
    Converts an OpenQuake IM string to a standard (gmhazard) IM string
    """
    # Define the regular expression pattern
    pattern = r'SA\((\d+)\.(\d+)\)'

    # Define the replacement pattern
    replacement = r'pSA_\1.\2'

    return re.sub(pattern, replacement, oq_im)

