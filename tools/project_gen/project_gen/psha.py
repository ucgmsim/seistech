import multiprocessing as mp
import os
from pathlib import Path
from typing import Sequence, Union

import yaml
import pandas as pd
import numpy as np
import celery

import gmhazard_calc as gc
from . import tasks
from . import utils


def gen_psha_project_data(project_dir: Path, n_procs: int = 1, use_mp: bool = True):
    """Computes the PSHA project data for the specified project directory"""
    project_id = project_dir.name
    results_dir = project_dir / "results"

    # Load the project definition config
    with open(project_dir / f"{project_id}.yaml", "r") as f:
        project_dict = yaml.safe_load(f)

    # Load the project parameters
    project_params = project_dict["project_parameters"]
    ims = gc.im.to_im_list(project_params["ims"])

    # TODO: Change this to only do the specified components
    im_components = [gc.im.IMComponent.RotD50]
    if "im_components" in project_params.keys():
        im_components = [
            gc.im.IMComponent[component]
            for component in project_params["im_components"]
        ]
        for im_component in im_components:
            if im_component != gc.im.IMComponent.RotD50:
                ims.extend(
                    [
                        gc.im.IM(im.im_type, im.period, im_component)
                        for im in ims
                        if im.is_pSA() or im.im_type == gc.im.IMType.PGA
                    ]
                )

    # Exceedances
    disagg_exceedances = [
        1 / cur_rp for cur_rp in project_params["disagg_return_periods"]
    ]
    uhs_exceedances = [1 / cur_rp for cur_rp in project_params["uhs_return_periods"]]

    # Load the ensemble
    ensemble_ffp = project_dict["ensemble_ffp"]
    ensemble = gc.gm_data.Ensemble(
        project_id, config_ffp=ensemble_ffp, use_im_data_cache=False
    )

    # Get the full list of station ids
    station_ids = project_params.get("location_ids")
    if station_ids is None:
        station_ids = [
            cur_site.station_name for cur_site in utils.get_site_infos(project_params)
        ]

    # Create the context & vs30 maps
    if use_mp:
        with mp.Pool(processes=n_procs) as p:
            p.starmap(
                generate_maps,
                [(ensemble, cur_station, results_dir) for cur_station in station_ids],
            )
    else:
        celery.group(
            tasks.generate_maps_task.s(
                project_id, ensemble_ffp, cur_station, str(results_dir)
            )
            for cur_station in station_ids
        )

    # Generate the station - IM combinations
    # Breaking calculations down into "smallest" chunks
    station_im_comb = [
        (cur_station, cur_im) for cur_station in station_ids for cur_im in ims
    ]

    # Run the hazard and disagg calculation
    if n_procs == 1:
        for cur_station, cur_im in station_im_comb:
            process_station_im(
                ensemble,
                cur_station,
                cur_im,
                disagg_exceedances,
                str(results_dir / cur_station / str(cur_im.component)),
            )
    elif use_mp:
        with mp.Pool(processes=n_procs) as p:
            p.starmap(
                process_station_im,
                [
                    (
                        ensemble,
                        cur_station,
                        cur_im,
                        disagg_exceedances,
                        str(results_dir / cur_station / str(cur_im.component)),
                    )
                    for cur_station, cur_im in station_im_comb
                ],
            )
    else:
        celery.group(
            tasks.process_station_im_task.s(
                project_id,
                ensemble_ffp,
                cur_station,
                cur_im,
                disagg_exceedances,
                str(results_dir / cur_station / str(cur_im.component)),
            )
            for cur_station, cur_im in station_im_comb
        )

    if len(uhs_exceedances) > 0:
        for cur_station in station_ids:
            cur_site_info = gc.site.get_site_from_name(ensemble, cur_station)

            # Computing UHS for each of the IM Components
            for im_component in im_components:
                cur_output_dir = results_dir / cur_station / str(im_component)

                if len(list(cur_output_dir.glob("uhs*"))) > 0:
                    print(
                        f"Skipping UHS generation for station {cur_station} - "
                        f"Component {im_component} as it already exists"
                    )
                    continue

                print(
                    f"Computing UHS for station {cur_station} - Component {im_component}"
                )

                # Compute & write UHS
                uhs_results = gc.uhs.run_ensemble_uhs(
                    ensemble,
                    cur_site_info,
                    np.asarray(uhs_exceedances),
                    n_procs=n_procs,
                    calc_percentiles=True,
                    im_component=im_component,
                )
                for cur_uhs_result in uhs_results:
                    cur_uhs_result.save(cur_output_dir)

                # Compute & write UHS NZS1170.5
                cur_uhs_nzs1170p5_dir = cur_output_dir / "uhs_nzs1170p5"
                cur_uhs_nzs1170p5_dir.mkdir(exist_ok=False, parents=False)
                uhs_nzs1170p5 = gc.uhs.run_nzs1170p5_uhs(
                    ensemble,
                    cur_site_info,
                    np.asarray(uhs_exceedances),
                    opt_nzs1170p5_args={"im_component": im_component},
                )
                for cur_uhs_nzs1170p5 in uhs_nzs1170p5:
                    cur_uhs_nzs1170p5.save(cur_uhs_nzs1170p5_dir, "uhs")

    if any(
        [
            cur_im_ens.im_data_type is gc.IMDataType.mixed
            for cur_im_ens in ensemble.im_ensembles
        ]
    ):
        print(f"Skipping Scenario generation as not all IM data is parametric")
        return

    # Generate the station - IMComponent combinations
    # Breaking calculations down into "smallest" chunks
    station_component_comb = [
        (cur_station, cur_component)
        for cur_station in station_ids
        for cur_component in im_components
    ]

    # Run the scenario calculations
    if use_mp:
        with mp.Pool(processes=n_procs) as p:
            p.starmap(
                process_station_scenarios,
                [
                    (
                        ensemble,
                        cur_station,
                        cur_component,
                        results_dir / cur_station / str(cur_component),
                    )
                    for cur_station, cur_component in station_component_comb
                ],
            )
    else:
        celery.group(
            tasks.process_station_component_task.s(
                ensemble,
                cur_station,
                cur_component,
                results_dir / cur_station / str(cur_component),
            )
            for cur_station, cur_component in station_im_comb
        )


def generate_maps(
    ensemble: gc.gm_data.Ensemble, station_name: str, results_dir: Union[str, Path]
):
    """Generates the context and vs30 map for the specified station

    Note: Existing maps are skipped
    """
    results_dir = Path(results_dir)

    site_info = gc.site.get_site_from_name(ensemble, station_name)
    output_dir = results_dir / station_name

    # Create the current output directory (if required)
    output_dir.mkdir(exist_ok=True, parents=False)

    # Generate the context maps
    if not (output_dir / "context_map_plot.png").exists():
        print(f"Generating context maps for station {site_info.station_name}")
        gc.plots.gmt_context(
            site_info.lon, site_info.lat, str(output_dir / "context_map_plot"),
        )
    else:
        print("Skipping context map generation as it already exists")

    # Generate the vs30 map
    if not (output_dir / "vs30_map_plot.png").exists():
        gc.plots.gmt_vs30(
            str(output_dir / "vs30_map_plot.png"),
            site_info.lon,
            site_info.lat,
            site_info.lon,
            site_info.lat,
            site_info.vs30,
            ensemble.station_ffp,
        )
    else:
        print("Skipping vs30 map generation as it already exists")


def process_station_scenarios(
    ensemble: gc.gm_data.Ensemble,
    station_name: str,
    im_component: gc.im.IMComponent,
    output_dir: Path,
):
    """Computes Scenarios for the specified station and IM Component"""
    print(f"Computing Scenarios for station {station_name} - Component {im_component}")

    site_info = gc.site.get_site_from_name(ensemble, station_name)

    # Compute and Write Scenario
    if (output_dir / gc.scenario.EnsembleScenarioResult.get_save_dir()).exists():
        print(
            f"Skipping scenario computation for station {station_name} "
            f"- IM component {im_component} as it already exists"
        )
    else:
        gc.scenario.run_ensemble_scenario(
            ensemble, site_info, im_component=im_component,
        ).save(output_dir)


def process_station_im(
    ensemble: gc.gm_data.Ensemble,
    station_name: str,
    im: gc.im.IM,
    disagg_exceedances: Sequence[float],
    output_dir: Union[str, Path],
):
    """Computes hazard, NZS1170.5 and disagg for the
    specified station and IM (and exceedances for disagg)

    Note: Existing results are skipped, this is done based on the
    existance of the output directory for the current station and IM
    E.g. for hazard it checks for the existance of hazard_{IM}
    This means that if a run fails, then the failed directory
    has to be cleaned up manually

    If a exceedance is out of the supported range
    for the specified station it is skipped (i.e. no results
    are generated and saved)
    """
    output_dir = Path(output_dir)
    print(f"\t{os.getpid()} - Processing station {station_name} and IM {im}")

    # Get the site
    site_info = gc.site.get_site_from_name(ensemble, station_name)

    # Compute & write hazard if needed
    ens_hazard = None
    if (output_dir / gc.hazard.EnsembleHazardResult.get_save_dir(im)).exists():
        print(
            f"\t{os.getpid()} - Skipping hazard computation for station {site_info.station_name} - "
            f"IM {im} - Component {im.component} as it already exists"
        )
    else:
        print(
            f"\t{os.getpid()} - Computing hazard for station"
            f" {site_info.station_name} - IM {im} - Component {im.component}"
        )
        ens_hazard = gc.hazard.run_ensemble_hazard(
            ensemble, site_info, im, calc_percentiles=True
        )
        ens_hazard.save(output_dir)

    # Compute & write NZS1170.5 if needed
    if im.im_type == gc.im.IMType.PGA or im.is_pSA():
        if (
            output_dir / gc.nz_code.nzs1170p5.NZS1170p5Result.get_save_dir(im, "hazard")
        ).exists():
            print(
                f"\t{os.getpid()} - Skipping NZS1170.5 computation for station {site_info.station_name} - "
                f"IM {im} - Component {im.component} as it already exists"
            )
        else:
            print(
                f"\t{os.getpid()} - Computing NZS1170.5 for station "
                f"{site_info.station_name} - IM {im} - Component {im.component}"
            )
            gc.nz_code.nzs1170p5.run_ensemble_nzs1170p5(ensemble, site_info, im).save(
                output_dir, "hazard"
            )

    # Compute & write NZTA hazard if needed
    if im.im_type == gc.im.IMType.PGA:
        if (output_dir / gc.nz_code.nzta_2018.NZTAResult.get_save_dir()).exists():
            print(
                f"\t{os.getpid()} - Skipping NZTA computation for station {site_info.station_name} "
                f"as it already exists"
            )
        else:
            print(
                f"\t{os.getpid()} - Computing NZTA for station {site_info.station_name}"
            )
            gc.nz_code.nzta_2018.run_ensemble_nzta(ensemble, site_info).save(output_dir)

    # Compute & write disagg for the different exceedances
    for cur_excd in disagg_exceedances:
        cur_rp = int(1.0 / cur_excd)

        if (
            output_dir
            / gc.disagg.EnsembleDisaggResult.get_save_dir(im, exceedance=cur_excd)
        ).exists():
            print(
                f"\t{os.getpid()} - Skipping disagg computation for station {site_info.station_name} - "
                f"IM {im} - Component {im.component} - Return period {cur_rp} as it already exists"
            )
        else:
            print(
                f"\t{os.getpid()} - Computing disagg for station {site_info.station_name} - "
                f"IM {im} - Component {im.component} - Return period {cur_rp}"
            )
            try:
                cur_disagg_data = gc.disagg.run_ensemble_disagg(
                    ensemble,
                    site_info,
                    im,
                    exceedance=cur_excd,
                    calc_mean_values=True,
                    hazard_result=ens_hazard,
                )
            except gc.exceptions.ExceedanceOutOfRangeError as ex:
                print(
                    f"\t{os.getpid()} - Failed to compute disagg for IM {ex.im} and exceedance {ex.exceedance} as the"
                    f"exceedance is outside of the computed hazard range for this site, skipping!"
                )
            else:
                cur_disagg_grid_data = gc.disagg.run_disagg_gridding(cur_disagg_data)

                # Save
                cur_disagg_data_dir = cur_disagg_data.save(output_dir)
                cur_disagg_grid_data.save(cur_disagg_data_dir, save_disagg_data=False)

                # Additional info for the table
                # Annual rec prob, magnitude and rrup (for disagg table)
                ruptures_df = ensemble.get_im_ensemble(im.im_type).rupture_df_id.loc[
                    cur_disagg_data.fault_disagg_id.index.values
                ]
                flt_dist_df = gc.site_source.get_distance_df(
                    ensemble.flt_ssddb_ffp, site_info
                )
                merged_df = pd.merge(
                    ruptures_df,
                    flt_dist_df,
                    how="left",
                    left_on="rupture_name",
                    right_index=True,
                )
                merged_df = merged_df.loc[
                    :, ["annual_rec_prob", "magnitude", "rupture_name", "rrup"]
                ]
                merged_df.to_csv(
                    cur_disagg_data_dir
                    / f"disagg_{im.file_format()}_{cur_rp}_metadata.csv",
                    index_label="record_id",
                )

                # Generate the disagg plots
                gc.plots.gmt_disagg(
                    str(
                        cur_disagg_data_dir / f"disagg_{im.file_format()}_{cur_rp}_src"
                    ),
                    cur_disagg_grid_data.to_dict(),
                    bin_type="src",
                )
                gc.plots.gmt_disagg(
                    str(
                        cur_disagg_data_dir / f"disagg_{im.file_format()}_{cur_rp}_eps"
                    ),
                    cur_disagg_grid_data.to_dict(),
                    bin_type="eps",
                )
                print(
                    f"\t{os.getpid()} - Completed disagg for station {site_info.station_name} - "
                    f"IM {im} - Component {im.component} - Return period {cur_rp}"
                )
