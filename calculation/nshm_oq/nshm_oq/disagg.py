import json
import shutil
from pathlib import Path

import pandas as pd
import gmhazard_common as common

from . import constants, utils


def setup_disagg_run(
    output_dir: Path,
    source_def_dir: Path,
    im_levels: dict[str, list[float] | float],
    n_logic_tree_branches: int,
    site_df: pd.DataFrame,
):
    """
    Sets up the hazard run directory

    Parameters
    ----------
    output_dir: Path
        Output directory for the hazard results
    source_def_dir: Path
        Source definition directory
    im_levels: dict[str, list[float] | float]
        The IM values for which to compute disagg per IM
    n_logic_tree_branches: int
        The number of logic tree branches to use
        If 0 the full logic tree is run,
        otherwise the number of branches specified is run
        (via Monte Carlo sampling)
    site_df: pd.DataFrame
        DataFrame containing the site information
        Required columns: lon, lat, vs30, z1p0, z2p5, backarc
    """
    run_dir, source_model_ffp = utils.setup_run_dir(
        output_dir, source_def_dir,site_df
    )

    # Copy the base_base_hazard ini file
    job_ini_ffp = run_dir / "disagg_job.ini"
    shutil.copy(
        constants.RESOURCE_DIR / constants.HAZARD_BASE_JOB_INI_FNAME, job_ini_ffp
    )

    # Update the disagg ini file
    with job_ini_ffp.open("r+") as f:
        base_job_ini = f.read(-1)
        base_job_ini = (
            base_job_ini.replace("$IM_LEVELS$", json.dumps(im_levels))
            .replace("$SOURCE_MODEL_XML_FFP$", source_model_ffp.name)
            .replace(
                "$N_RELS$",
                str(0) if n_logic_tree_branches is None else str(n_logic_tree_branches),
            )
        )
        f.seek(0)
        f.write(base_job_ini)

    return job_ini_ffp
