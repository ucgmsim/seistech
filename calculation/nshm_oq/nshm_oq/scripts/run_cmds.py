import json
import shutil
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from nshm_oq import constants, utils
import gmhazard_common as common

app = typer.Typer()

RESOURCE_DIR = Path(__file__).parent.parent / "resources"


@app.command("run-hazard")
def run_hazard(
    lon: Annotated[float, typer.Argument(help="Longitude of the site")],
    lat: Annotated[
        float,
        typer.Argument(
            help="Latitude of the site. To enter a negative number put -- in front, e.g. -- -34.5",
        ),
    ],
    vs30: Annotated[float, typer.Argument(help="Vs30 at the site")],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory in which the site dir will be created"),
    ],
    source_def_dir: Annotated[
        Path, typer.Argument(help="Source definitions directory")
    ],
    z1p0: Annotated[
        float,
        typer.Argument(
            help="Z1.0 at the site of interest, in metres. "
            "If not specified, then it is estimated based on "
            "the Vs30 correlation function used in the NSHM results."
        ),
    ] = None,
    z2p5: Annotated[
        float,
        typer.Argument(
            help="Z2.5 at the site of interest, in kilometres. "
            "If not specified, then it is estimated based on "
            "the Vs30 correlation function used in the NSHM results."
        ),
    ] = None,
    sa_periods: Annotated[
        constants.SAPeriodOptions, typer.Option(help="The number of pSA periods to use")
    ] = constants.SAPeriodOptions.extended,
    im_levels: Annotated[
        constants.IMLevelOptions, typer.Option(help="The number of IM levels to use")
    ] = constants.IMLevelOptions.extended,
    n_logic_tree_branches: Annotated[
        int,
        typer.Option(
            help="If specified, then MC sampling is used to run the specified number of logic tree branches."
            "If None (default) then the full logic tree is evaluated",
        ),
    ] = None,
):
    # Create the run dir
    (run_dir := output_dir / "run_dir").mkdir(exist_ok=False, parents=False)

    # Get the IMs & Levels
    sa_periods = (
        constants.NSHM_SA_PERIODS
        if sa_periods is constants.SAPeriodOptions.nshm
        else constants.EXTENDED_SA_PERIODS
    )
    ims = [common.im.IM.from_str("PGA")] + [
        common.im.IM(common.im.IMType.pSA, cur_period) for cur_period in sa_periods
    ]

    im_levels = {
        utils.to_oq_im_string(cur_im): (
            constants.NSHM_IM_LEVELS
            if im_levels is constants.IMLevelOptions.nshm
            else list(
                common.hazard.get_im_levels(cur_im, constants.EXTENDEND_N_IM_LEVELS)
            )
        )
        for cur_im in ims
    }

    # Copy the base_base_hazard ini file & source model xml
    job_ini_ffp = run_dir / "hazard_job.ini"
    shutil.copy(RESOURCE_DIR / constants.HAZARD_BASE_JOB_INI_FNAME, job_ini_ffp)
    source_model_ffp = run_dir / "source_model.xml"
    shutil.copy(RESOURCE_DIR / constants.BASE_SOURCE_MODEL_FNAME, source_model_ffp)

    # Copy the GMM logic tree
    shutil.copy(RESOURCE_DIR / constants.GMM_LOGIC_TREE_FNAME, run_dir / constants.GMM_LOGIC_TREE_FNAME)

    # Link the source definitions directory
    (local_source_def_dir := run_dir / "source_definitions").symlink_to(source_def_dir)

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

    # Create the site.csv file
    site_df = pd.Series(
        dict(
            lon=lon,
            lat=lat,
            vs30=vs30,
            z1pt0=utils.calculate_z1pt0(vs30) if z1p0 is None else z1p0,
            z2pt5=utils.calculate_z2pt5_ngaw2(vs30) if z2p5 is None else z2p5,
            backarc=int(
                utils.get_backarc_mask(
                    RESOURCE_DIR / constants.BACKARC_POLYGON_FNAME,
                    np.asarray([lon, lat])[None, :],
                )[0]
            ),
        )
    ).to_frame().T.to_csv(run_dir / "sites.csv", index=False)

    print("wtf")


if __name__ == "__main__":
    app()
