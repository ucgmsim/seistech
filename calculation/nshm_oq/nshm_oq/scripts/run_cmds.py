import time
import json
import shutil
from pathlib import Path
from typing import Annotated
from openquake.commonlib import logs
from openquake.calculators.base import calculators

import numpy as np
import pandas as pd
import typer

from nshm_oq import constants, utils, hazard

app = typer.Typer()




@app.command("run-single-site-hazard")
def run_single_site_hazard(
    lon: Annotated[float, typer.Argument(help="Longitude of the site")],
    lat: Annotated[
        float,
        typer.Argument(
            help="Latitude of the site. To enter a negative number"
                 " put -- in front, e.g. -- -34.5",
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
    pSA_periods_option: Annotated[
        constants.SAPeriodOptions, typer.Option(help="The pSA period set to use")
    ] = constants.SAPeriodOptions.extended,
    im_levels_option: Annotated[
        constants.IMLevelOptions, typer.Option(help="The number of IM levels to use")
    ] = constants.IMLevelOptions.extended,
    n_branches: Annotated[
        int,
        typer.Option(
            help="If specified, then MC sampling is used to run the specified "
            "number of logic tree branches (via Monte Carlo sampling). "
            "If None (default) then the full logic tree is evaluated",
        ),
    ] = None,
):
    # Create the site.csv file
    site_df = (
        pd.Series(
            dict(
                lon=lon,
                lat=lat,
                vs30=vs30,
                z1pt0=utils.calculate_z1pt0(vs30) if z1p0 is None else z1p0,
                z2pt5=utils.calculate_z2pt5_ngaw2(vs30) if z2p5 is None else z2p5,
                backarc=int(
                    utils.get_backarc_mask(
                        constants.RESOURCE_DIR / constants.BACKARC_POLYGON_FNAME,
                        np.asarray([lon, lat])[None, :],
                    )[0]
                ),
            )
        )
        .to_frame()
        .T
    )

    job_ini_ffp = hazard.setup_hazard_run(
        output_dir,
        source_def_dir,
        im_levels_option,
        pSA_periods_option,
        n_branches,
        site_df,
    )

    start_time = time.time()
    with logs.init(job_ini_ffp) as log:
        print(f"OQ Calculation ID: {log.calc_id}")
        oq = log.get_oqparam()
        oq.use_rates = True
        oq.max_potential_paths = 100_000
        calc = calculators(oq, log.calc_id)
        calc.run()
    print(f"Took {time.time() - start_time} to run the OQ hazard calculation")



@app.command("run-single-site-disagg")
def run_single_site_disagg(
        lon: Annotated[float, typer.Argument(help="Longitude of the site")],
        lat: Annotated[
            float,
            typer.Argument(
                help="Latitude of the site. To enter a negative number"
                     " put -- in front, e.g. -- -34.5",
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
        pSA_periods_option: Annotated[
            constants.SAPeriodOptions, typer.Option(help="The pSA period set to use")
        ] = constants.SAPeriodOptions.extended,
        im_levels_option: Annotated[
            constants.IMLevelOptions, typer.Option(help="The number of IM levels to use")
        ] = constants.IMLevelOptions.extended,
        n_branches: Annotated[
            int,
            typer.Option(
                help="If specified, then MC sampling is used to run the specified "
                     "number of logic tree branches (via Monte Carlo sampling). "
                     "If None (default) then the full logic tree is evaluated",
            ),
        ] = None,
):
    pass

if __name__ == "__main__":
    app()
