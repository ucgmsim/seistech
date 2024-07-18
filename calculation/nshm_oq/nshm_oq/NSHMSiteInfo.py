import json
from pathlib import Path

import gmhazard_common as common


class NSHMSiteInfo(common.SiteInfo):
    """
    NZ NSHM specific site information class

    Parameters
    ----------
    name: str
    lat: float
    lon: float
    vs30: float
        The vs30 value at the site
    backarc: bool
        True if the site is in the backarc region
        False otherwise
    z1p0: float, optional
        Depth of Vs=1000 in metres at the site
    z2p5: float, optional
        Depth of Vs=2500 in kilometres at the site
    """

    def __init__(
        self,
        name: str,
        lat: float,
        lon: float,
        vs30: float,
        backarc: bool,
        z1p0: float = None,
        z2p5: float = None,
    ):
        super().__init__(name, lat, lon, vs30, z1p0, z2p5)
        self._backarc = backarc

    @property
    def backarc(self):
        return self._backarc

    def save(self, save_dir: Path, metadata: dict = None):
        metadata = {} if metadata is None else metadata
        metadata["backarc"] = self._backarc
        super().save(save_dir, metadata)

    @classmethod
    def load(cls, data_dir: Path):
        with open(data_dir / cls.SITE_DATA_FN, "r") as f:
            data = json.load(f)

        return cls(
            data["name"],
            data["lat"],
            data["lon"],
            data["vs30"],
            data["backarc"],
            data["z1p0"],
            data["z2p5"],
        )
