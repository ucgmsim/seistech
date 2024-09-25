import time
from pathlib import Path
from typing import Annotated
from openquake.commonlib import logs
from openquake.calculators.base import calculators


import numpy as np
import pandas as pd
import typer

import nshm_oq
import gmhazard_common as common


app = typer.Typer()


@app.command("run-single-site-hazard")
def run_single_site_hazard(
    name: Annotated[str, typer.Argument(help="Name of the site")],
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
        typer.Option(
            help="Z1.0 at the site of interest, in metres. "
            "If not specified, then it is estimated based on "
            "the Vs30 correlation function used in the NSHM results."
        ),
    ] = None,
    z2p5: Annotated[
        float,
        typer.Option(
            help="Z2.5 at the site of interest, in kilometres. "
            "If not specified, then it is estimated based on "
            "the Vs30 correlation function used in the NSHM results."
        ),
    ] = None,
    pSA_periods_option: Annotated[
        nshm_oq.constants.SAPeriodOptions,
        typer.Option(help="The pSA period set to use"),
    ] = nshm_oq.constants.SAPeriodOptions.extended,
    im_levels_option: Annotated[
        nshm_oq.constants.IMLevelOptions,
        typer.Option(help="The number of IM levels to use"),
    ] = nshm_oq.constants.IMLevelOptions.extended,
    n_branches: Annotated[
        int,
        typer.Option(
            help="If specified, then MC sampling is used to run the specified "
            "number of logic tree branches (via Monte Carlo sampling). "
            "If None (default) then the full logic tree is evaluated",
        ),
    ] = None,
):
    """Runs single site hazard calculation for NZ NSHM using OQ"""

    # Create the site object
    site = nshm_oq.NSHMSiteInfo(
        name,
        lat,
        lon,
        vs30,
        bool(
            nshm_oq.utils.get_backarc_mask(
                nshm_oq.constants.RESOURCE_DIR
                / nshm_oq.constants.BACKARC_POLYGON_FNAME,
                np.asarray([lon, lat])[None, :],
            )[0]
        ),
        nshm_oq.utils.calculate_z1pt0(vs30) if z1p0 is None else z1p0,
        nshm_oq.utils.calculate_z2pt5_ngaw2(vs30) if z2p5 is None else z2p5,
    )

    # Create the site.csv file
    site_df = (
        pd.Series(
            dict(
                lon=lon,
                lat=lat,
                vs30=vs30,
                z1pt0=site.z1p0,
                z2pt5=site.z2p5,
                backarc=int(site.backarc),
            )
        )
        .to_frame()
        .T
    )

    # Setup
    job_ini_ffp, ims = nshm_oq.hazard.setup_hazard_run(
        output_dir,
        source_def_dir,
        im_levels_option,
        pSA_periods_option,
        n_branches,
        site_df,
    )

    # Run OQ
    start_time = time.time()
    with logs.init(job_ini_ffp) as log:
        calc_id = log.calc_id
        print(f"OQ Calculation ID: {log.calc_id}")
        oq = log.get_oqparam()
        oq.use_rates = True
        oq.max_potential_paths = 1_000_000
        calc = calculators(oq, log.calc_id)
        calc.run()
    print(f"Took {time.time() - start_time} to run the OQ hazard calculation")

    hazard_results = nshm_oq.hazard.get_single_site_hazard_results(calc_id, site)

    # Write results
    for cur_im, cur_hazard_result in hazard_results.items():
        cur_hazard_result.save(output_dir / cur_im.file_format)


@app.command("run-single-site-disagg")
def run_single_site_disagg(
    output_dir: Annotated[Path, typer.Argument(help="Directory to save the results")],
    hazard_results_dir: Annotated[
        Path, typer.Argument(help="Directory containing the hazard results")
    ],
    source_def_dir: Annotated[
        Path, typer.Argument(help="Source definitions directory")
    ],
    rp: Annotated[int, typer.Argument(help="Return period to run disagg for")],
    n_branches: Annotated[
        int,
        typer.Option(
            help="If specified, then MC sampling is used to run the specified "
            "number of logic tree branches (via Monte Carlo sampling). "
            "If None (default) then the full logic tree is evaluated",
        ),
    ] = None,
):
    """Runs single site disagg based on existing hazard results"""
    # Load the hazard results
    # Bit of a hack, but once we switch to project approach
    # this won't be an issue anymore
    hazard_results = {}
    site = None
    for cur_dir in hazard_results_dir.iterdir():
        if cur_dir.is_dir() and (
            cur_dir.stem == "PGA" or cur_dir.stem.startswith("pSA")
        ):
            hazard_results[common.IM.from_str(cur_dir.stem)] = (
                common.hazard.HazardResult.load(cur_dir)
            )

            if site is None:
                site = nshm_oq.NSHMSiteInfo.load(cur_dir)

    ims = list(hazard_results.keys())

    # # Compute the im-levels for each IM and RP
    # rps = sorted(rps)
    # excd_im_levels = {}
    # for cur_im in ims:
    #     excd_im_levels[cur_im] = []
    #     for cur_rp in rps:
    #         cur_excd = common.utils.rp_to_prob(cur_rp)
    #         excd_im_levels[cur_im].append(
    #             common.hazard.exceedance_to_im(cur_excd, hazard_results[cur_im])
    #         )

    # Compute the im-levels for each IM
    excd_im_levels = {}
    for cur_im in ims:
        cur_excd = common.utils.rp_to_prob(rp)
        excd_im_levels[cur_im] = (
            common.hazard.exceedance_to_im(cur_excd, hazard_results[cur_im])
        )

    # Create the site.csv file
    site_df = (
        pd.Series(
            dict(
                lon=site.lon,
                lat=site.lat,
                vs30=site.vs30,
                z1pt0=site.z1p0,
                z2pt5=site.z2p5,
                backarc=site.backarc,
            )
        )
        .to_frame()
        .T
    )

    oq_excd_im_levels = {cur_im.oq_str: cur_levels for cur_im, cur_levels in excd_im_levels.items()}
    job_ini_ffp = nshm_oq.disagg.setup_disagg_run(
        output_dir,
        source_def_dir,
        oq_excd_im_levels,
        n_branches,
        site_df,
    )

    start_time = time.time()
    with logs.init(job_ini_ffp) as log:
        print(f"OQ Calculation ID: {log.calc_id}")
        oq = log.get_oqparam()
        oq.use_rates = True
        oq.max_potential_paths = 1_000_000
        calc = calculators(oq, log.calc_id)
        calc.run()
    print(f"Took {time.time() - start_time} to run the OQ hazard calculation")


if __name__ == "__main__":
    app()
