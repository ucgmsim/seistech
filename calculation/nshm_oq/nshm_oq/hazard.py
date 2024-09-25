import shutil
import json
from pathlib import Path

import pandas as pd
import numpy as np
import gmhazard_common as common
from openquake.calculators import extract
from openquake.commonlib import datastore


from . import constants, utils
from .NSHMSiteInfo import NSHMSiteInfo


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
    run_dir, source_model_ffp = utils.setup_run_dir(output_dir, source_def_dir, site_df)

    # Get the IMs & Levels
    sa_periods = (
        constants.NSHM_SA_PERIODS
        if pSA_periods_option is constants.SAPeriodOptions.nshm
        else constants.EXTENDED_SA_PERIODS
    )
    ims = [common.IM.from_str("PGA")] + [
        common.IM(common.IMType.pSA, cur_period) for cur_period in sa_periods
    ]

    # Copy the base_base_hazard ini file
    job_ini_ffp = run_dir / "hazard_job.ini"
    shutil.copy(
        constants.RESOURCE_DIR / constants.HAZARD_BASE_JOB_INI_FNAME, job_ini_ffp
    )

    im_levels = {
        cur_im.oq_str: (
            constants.NSHM_IM_LEVELS
            if im_levels_option is constants.IMLevelOptions.nshm
            else list(
                common.hazard.get_im_levels(cur_im, constants.EXTENDEND_N_IM_LEVELS)
            )
        )
        for cur_im in ims
    }

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

    return job_ini_ffp, ims


def get_single_site_hazard_results(calc_id: int, site: NSHMSiteInfo):
    """
    Get the hazard results for the given calculation id and site

    Parameters
    ----------
    calc_id: int
        The OQ calculation id
    site: NSHMSiteInfo
        The site information

    Returns
    -------
    LabelledDataArray:
        The hazard results
    """
    # Get the available IMs
    with datastore.read(calc_id) as ds:
        oq_params = ds["oqparam"]
    im_levels: dict[str, np.ndarray] = dict(oq_params.imtls)
    ims = list(im_levels.keys())

    result_dict = {}
    with extract.Extractor(calc_id) as ex:
        for cur_oq_im_str in ims:
            cur_oq_result = ex.get(f"hcurves?kind=stats&imt={cur_oq_im_str}")

            cur_im = common.IM.from_oq_str(cur_oq_im_str)
            cur_hazard_result = common.HazardResult(
                cur_im,
                site,
                pd.DataFrame(
                    data=np.stack(
                        [
                            cur_oq_result[cur_stats][0, 0, :]
                            for cur_stats in cur_oq_result.kind
                        ],
                        axis=1,
                    ),
                    index=im_levels[cur_oq_im_str],
                    columns=cur_oq_result.kind,
                ),
            )

            result_dict[cur_im] = cur_hazard_result

    return result_dict
