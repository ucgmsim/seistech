from pathlib import Path
import json

import geojson
import numpy as np
from turfpy import measurement

import gmhazard_common as common


def to_oq_im_string(im: common.im.IM):
    """Converts an IM to an OpenQuake IM string"""
    if im.is_pSA():
        return f"SA({im.period})"
    return str(im)


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
            for cur_point in measurement.points_within_polygon(points, polygon)["features"]
        ],
    )
    backarc_mask = np.zeros(shape=locs.shape[0], dtype=bool)
    backarc_mask[backarc_ind] = True

    return backarc_mask


def calculate_z1pt0(vs30):
    """
    Reads an array of vs30 values (in m/s) and
    returns the depth to the 1.0 km/s velocity horizon (in m)
    Ref: Chiou & Youngs (2014) California model
    :param vs30: the shear wave velocity (in m/s) at a depth of 30m
    """
    c1 = 571 ** 4.0
    c2 = 1360.0 ** 4.0
    return np.exp((-7.15 / 4.0) * np.log((vs30 ** 4.0 + c1) / (c2 + c1)))


def calculate_z2pt5_ngaw2(vs30):
    """
    Reads an array of vs30 values (in m/s) and
    returns the depth to the 2.5 km/s velocity horizon (in km)
    Ref: Campbell, K.W. & Bozorgnia, Y., 2014.
    'NGA-West2 ground motion model for the average horizontal components of
    PGA, PGV, and 5pct damped linear acceleration response spectra.'
    Earthquake Spectra, 30(3), pp.1087–1114.

    :param vs30: the shear wave velocity (in m/s) at a depth of 30 m
    """
    c1 = 7.089
    c2 = -1.144
    z2pt5 = np.exp(c1 + np.log(vs30) * c2)
    return z2pt5