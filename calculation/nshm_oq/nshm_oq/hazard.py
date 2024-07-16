import shutil
import json
from pathlib import Path

import pandas as pd
import gmhazard_common as common

from . import constants, utils


def setup_hazard_run(
    output_dir: Path,
    source_def_dir: Path,
    im_levels_option: constants.IMLevelOptions,
    pSA_periods_option: constants.SAPeriodOptions,
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
    im_levels_option: IMLevelOptions
        The IM levels to use
    pSA_periods_option: SAPeriodOptions
        The pSA periods to use
    n_logic_tree_branches: int
        The number of logic tree branches to use
        If 0 the full logic tree is run,
        otherwise the number of branches specified is run
        (via Monte Carlo sampling)
    site_df: pd.DataFrame
        DataFrame containing the site information
        Required columns: lon, lat, vs30, z1p0, z2p5, backarc
    """
    # Create the run dir
    (run_dir := output_dir / "run_dir").mkdir(exist_ok=False, parents=False)

    # Link the source definitions directory
    (local_source_def_dir := run_dir / "source_definitions").symlink_to(source_def_dir)

    # Get the IMs & Levels
    sa_periods = (
        constants.NSHM_SA_PERIODS
        if pSA_periods_option is constants.SAPeriodOptions.nshm
        else constants.EXTENDED_SA_PERIODS
    )
    ims = [common.im.IM.from_str("PGA")] + [
        common.im.IM(common.im.IMType.pSA, cur_period) for cur_period in sa_periods
    ]

    im_levels = {
        utils.to_oq_im_string(cur_im): (
            constants.NSHM_IM_LEVELS
            if im_levels_option is constants.IMLevelOptions.nshm
            else list(
                common.hazard.get_im_levels(cur_im, constants.EXTENDEND_N_IM_LEVELS)
            )
        )
        for cur_im in ims
    }

    # Copy the base_base_hazard ini file & source model xml
    job_ini_ffp = run_dir / "hazard_job.ini"
    shutil.copy(constants.RESOURCE_DIR / constants.HAZARD_BASE_JOB_INI_FNAME, job_ini_ffp)
    source_model_ffp = run_dir / "source_model.xml"
    shutil.copy(constants.RESOURCE_DIR / constants.BASE_SOURCE_MODEL_FNAME, source_model_ffp)

    # Copy the GMM logic tree
    shutil.copy(
        constants.RESOURCE_DIR / constants.GMM_LOGIC_TREE_FNAME,
        run_dir / constants.GMM_LOGIC_TREE_FNAME,
    )

    # Update the hazard ini file
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

    # Update the source model xml file
    with source_model_ffp.open("r+") as f:
        base_source_model = f.read(-1)
        base_source_model = base_source_model.replace(
            "$SOURCE_DIR$", local_source_def_dir.name
        )
        f.seek(0)
        f.write(base_source_model)

    # Write the site file
    site_df.to_csv(run_dir / "sites.csv", index=False)



    return job_ini_ffp
