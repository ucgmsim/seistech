import json
from pathlib import Path

import numpy as np


class SiteInfo:
    """
    Contains information for a single site

    Properties
    ----------
    name: str
    lat: float
    lon: float
    vs30: float
        The vs30 value at the site
    z1p0: float, optional
        Depth of Vs=1000 in metres at the site
    z2p5: float, optional
        Depth of Vs=2500 in kilometres at the site
    """

    SITE_DATA_FN = "site.json"

    def __init__(
        self,
        name: str,
        lat: float,
        lon: float,
        vs30: float,
        z1p0: float = None,
        z2p5: float = None,
    ):
        self._station_name = name
        self._lat, self._lon = lat, lon
        self._vs30 = vs30
        self._z1p0 = z1p0
        self._z2p5 = z2p5

    @property
    def station_name(self):
        return self._station_name

    @property
    def lat(self):
        return self._lat

    @property
    def lon(self):
        return self._lon

    @property
    def vs30(self):
        return self._vs30

    @property
    def z1p0(self):
        return self._z1p0

    @property
    def z2p5(self):
        return self._z2p5

    def __str__(self):
        return f"{self.lon}_{self.lat}_{self.vs30}_{self._z1p0}_{self._z2p5}"

    def __repr__(self):
        return f'SiteInfo("{self.station_name}", {self.lat}, {self.lon}, {self.vs30}, {self.z1p0}, {self.z2p5})'

    def save(self, save_dir: Path, metadata: dict = None):
        """Saves the site information to the specified directory"""
        save_dict = {
            "name": self._station_name,
            "lat": float(self._lat),
            "lon": float(self._lon),
            "vs30": float(self._vs30),
            "z1p0": (
                None
                if self._z1p0 is None or np.isnan(self._z1p0)
                else float(self._z1p0)
            ),
            "z2p5": (
                None
                if self._z2p5 is None or np.isnan(self._z2p5)
                else float(self._z2p5)
            ),
        }
        if metadata is not None:
            save_dict = {**save_dict, **metadata}

        with open(save_dir / self.SITE_DATA_FN, "w") as f:
            json.dump(
                save_dict,
                f,
            )

    @classmethod
    def load(cls, data_dir: Path):
        """Loads the hazard result from the specified directory"""
        with open(data_dir / cls.SITE_DATA_FN, "r") as f:
            data = json.load(f)

        return cls(
            data["name"],
            data["lat"],
            data["lon"],
            data["vs30"],
            data["z1p0"],
            data["z2p5"],
        )
