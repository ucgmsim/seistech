import json
from pathlib import Path

import pandas as pd
import numpy as np

from . import IM
from .SiteInfo import SiteInfo


class HazardResult:
    """
    Hazard result data for a single
    site and IM


    Attributes
    ----------
    im : IM
        IM Object for the hazard curve.
    site : SiteInfo
        The Site of this result
    hazard_stats : pd.DataFrame
        Hazard statistics
        Index: IM levels
        Columns: Statistics, such as mean,
            quantiles, ds, fault, etc
            Required: ["mean"]
    """

    # File names for saving
    METADATA_FN = "metadata.json"
    HAZARD_STATS_FN = "hazard_stats.parquet"

    def __init__(self, im: IM, site: SiteInfo, hazard_stats: pd.DataFrame):
        self.im = im
        self.site = site
        self.hazard_stats = hazard_stats

    @property
    def mean_hazard(self):
        """Mean annual exceedance rates"""
        return self.hazard_stats["mean"]

    @property
    def im_levels(self):
        """The IM levels for which the hazard was calculated"""
        return self.hazard_stats.index.values.astype(float)

    def __repr__(self):
        return f"HazardResult({self.site}, {self.im})"

    def save(self, out_dir: Path, metadata: dict = None):
        """
        Saves the result to the specified directory

        Parameters
        ----------
        out_dir: Path
        metadata: dict, optional
            Additional metadata to save with the hazard result
        """
        out_dir.mkdir(parents=False, exist_ok=True)
        self.hazard_stats.to_parquet(out_dir / self.HAZARD_STATS_FN)

        self.site.save(out_dir)

        # Save the metadata
        metadata = metadata if metadata is not None else {}
        with open(out_dir / self.METADATA_FN, "w") as f:
            json.dump({**{"im": str(self.im)}, **metadata}, f)

    @classmethod
    def load(cls, data_dir: Path):
        """
        Load the HazardResult data from the specified directory

        Parameters
        ----------
        data_dir: Path

        Returns
        -------
        HazardResult
        """
        with open(data_dir / cls.METADATA_FN, "r") as f:
            metadata = json.load(f)

        return cls(
            IM.from_str(metadata["im"]),
            SiteInfo.load(data_dir),
            pd.read_parquet(data_dir / cls.HAZARD_STATS_FN),
        )
