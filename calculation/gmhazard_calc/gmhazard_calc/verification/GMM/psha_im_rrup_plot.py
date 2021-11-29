import pathlib
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import time
from collections.abc import Iterable

import constants as const
from empirical.util import empirical_factory
from empirical.util.classdef import Site, Fault, TectType


def init_setup(mag_dict: Dict, vs30_values: List, psa_periods: List):
    """Create nested dictionaries for fault and result
    based on a different im, vs30, mag and model
    also period if im is pSA

    Parameters
    ----------
    mag_dict: Dict
        List of Magnitudes
    vs30_values: List
        List of Vs30s
    psa_periods: List
        List of pSA Periods
    """
    faults = {}
    result_dict = {}
    for tect_type, im_models in const.MODELS_DICT.items():
        faults[tect_type] = {}
        result_dict[tect_type] = {}
        for im, models in im_models.items():
            result_dict[tect_type][im] = {}
            for vs30 in vs30_values:
                faults[tect_type][vs30] = {}
                result_dict[tect_type][im][vs30] = {}
                for mag in mag_dict[tect_type]:
                    faults[tect_type][vs30][mag] = []
                    result_dict[tect_type][im][vs30][mag] = {}
                    if im == const.PSA_IM_NAME:
                        for psa_period in psa_periods:
                            result_dict[tect_type][im][vs30][mag][psa_period] = {}
                            for model in models:
                                result_dict[tect_type][im][vs30][mag][psa_period][
                                    model
                                ] = []
                    else:
                        for model in models:
                            result_dict[tect_type][im][vs30][mag][model] = []

    return faults, result_dict


def get_sites(vs30_values: List, rrup_values: Dict):
    """Creates a dictionary
    pair of a different Vs30 and Sites

    Parameters
    ----------
    vs30_values: List
        list of Vs30s
    rrup_values: Dict
        dictionary of Rrups np.ndarray
    """
    # sites = {vs30: [] for vs30 in vs30_values}
    sites = {}

    for tect_type in const.CONST_FAULT_PARAMS.keys():
        sites[tect_type] = {}
        for vs30 in vs30_values:
            sites[tect_type][vs30] = []
            for rrup in rrup_values.get(tect_type):
                sites[tect_type][vs30].append(
                    Site(
                        rrup=rrup,
                        rjb=rrup,
                        rx=rrup,
                        ry=rrup,
                        vs30=vs30,
                        **const.CONST_SITE_PARAMS,
                    )
                )

    return sites


def get_faults(vs30_values: List, mag_dict: Dict, faults: Dict):
    """Creates a dictionary
     based on a different Vs30, Magnitude and Faults

    Parameters
    ----------
    vs30_values: List
        list of Vs30s
    mag_dict: Dict
        Dictionary with a different Mw lists for a different tectonic type
    faults: Dict
        nested dictionary
    """
    for tect_type in const.MODELS_DICT.keys():
        for vs30 in vs30_values:
            for mag in mag_dict[tect_type]:
                faults[tect_type][vs30][mag].append(
                    Fault(
                        Mw=mag,
                        tect_type=TectType[tect_type],
                        **const.CONST_FAULT_PARAMS[tect_type],
                    )
                )

    return faults


def get_computed_gmms(
    vs30_values: List,
    sites: Dict,
    mag_dict: Dict,
    faults: Dict,
    result_dict: Dict,
    psa_periods: List,
):
    """Get computed GMMs

    Parameters
    ----------
    vs30_values: List
        list of Vs30s
    sites: Dict
        nested dictionary of Sites with a different Vs30
    mag_dict: Dict
        Dictionary with a different Mw lists for a different tectonic type
    faults: Dict
        nested dictionary of Faults with a different Vs30 and Magnitude
    result_dict: Dict
        nested dictionary with a different im, Vs30 and Magnitude
    psa_periods: List
        list of Periods
    """
    for tect_type, im_models in const.MODELS_DICT.items():
        for im, models in im_models.items():
            for vs30 in vs30_values:
                for site in sites[tect_type][vs30]:
                    for mag in mag_dict[tect_type]:
                        for fault in faults[tect_type][vs30][mag]:
                            for model in models:
                                if im == const.PSA_IM_NAME:
                                    for psa_period in psa_periods:
                                        result = empirical_factory.compute_gmm(
                                            fault,
                                            site,
                                            empirical_factory.GMM[model],
                                            im,
                                            [psa_period],
                                        )
                                        # result is either
                                        # list of tuples or list
                                        result_dict[tect_type][im][vs30][mag][
                                            psa_period
                                        ][model].append(
                                            result[0][0]
                                            if isinstance(result, List)
                                            else result[0]
                                        )
                                else:
                                    result = empirical_factory.compute_gmm(
                                        fault, site, empirical_factory.GMM[model], im,
                                    )
                                    # result is always tuple
                                    # For Meta
                                    # result_dict[tect_type][im][vs30][mag][model].append(
                                    #     result[0][0]
                                    #     if isinstance(result, Iterable)
                                    #     else result[0]
                                    # )
                                    result_dict[tect_type][im][vs30][mag][model].append(
                                        result[0]
                                    )

    return result_dict


def plot_im_rrup(
    vs30_values: List,
    mag_dict: Dict,
    rrup_values: Dict,
    result_dict: Dict,
    plot_directory: pathlib.PosixPath,
):
    """Plots for IM(no pSA) versus Rrup

    Parameters
    ----------
    vs30_values: List
        list of Vs30s
    mag_dict: Dict
        Dictionary with a different Mw lists for a different tectonic type
    rrup_values: Dict
        dictionary of Rrups np.ndarray
    result_dict: Dict
        nested dictionary with a different im, Vs30 and Magnitude
    plot_directory: pathlib.PosixPath
        absolute path for a directory to store plot image
    """
    for tect_type, im_models in const.MODELS_DICT.items():
        for im, models in im_models.items():
            if im != const.PSA_IM_NAME:
                x_position = 0
                fig, ax = plt.subplots(
                    len(vs30_values),
                    len(mag_dict[tect_type]),
                    figsize=(18, 13.5),
                    dpi=300,
                )
                for vs30 in vs30_values:
                    y_position = 0
                    for mag in mag_dict[tect_type]:
                        color_index = 0
                        for model in models:
                            # To match the color with global version
                            if model.endswith("NZ"):
                                color_index -= 1
                            ax[x_position, y_position].plot(
                                rrup_values.get(tect_type),
                                result_dict[tect_type][im][vs30][mag][model],
                                label=model,
                                color=const.DEFAULT_LABEL_COLOR[color_index],
                                linestyle="dashed" if model.endswith("NZ") else "solid",
                            )
                            color_index += 1

                        ax[x_position, y_position].set_title(
                            f"{im} versus Rrup - Mw{mag}, Vs30-{vs30}"
                        )

                        ax[x_position, y_position].legend(models)
                        ax[x_position, y_position].xaxis.set_label_text("Rrup [km]")
                        ax[x_position, y_position].yaxis.set_label_text(f"{im}")
                        ax[x_position, y_position].set_xscale("log")
                        ax[x_position, y_position].set_yscale("log")
                        ax[x_position, y_position].set_xlim(left=10)
                        ax[x_position, y_position].xaxis.grid(
                            True, which="both", linestyle="dotted"
                        )
                        ax[x_position, y_position].yaxis.grid(
                            True, which="both", linestyle="dotted"
                        )
                        y_position += 1
                    x_position += 1

                fig.tight_layout()
                plt.savefig(f"{plot_directory}/{tect_type}_{im}.png")
                plt.close()


def plot_psa_rrup(
    vs30_values: List,
    mag_dict: Dict,
    psa_periods: List,
    rrup_values: Dict,
    result_dict: Dict,
    plot_directory: pathlib.PosixPath,
):
    """Plots for pSA versus Rrup

    Parameters
    ----------
    vs30_values: List
        list of Vs30s
    mag_dict: Dict
        Dictionary with a different Mw lists for a different tectonic type
    psa_periods: List
        list of Periods
    rrup_values: Dict
        dictionary of Rrups np.ndarray
    result_dict: Dict
        nested dictionary with a different im, Vs30 and Magnitude
    plot_directory: pathlib.PosixPath
        absolute path for a directory to store plot image
    """
    for tect_type, im_models in const.MODELS_DICT.items():
        for psa_period in psa_periods:
            x_position = 0
            fig, ax = plt.subplots(
                len(vs30_values), len(mag_dict[tect_type]), figsize=(18, 13.5), dpi=300
            )
            im_label = f"SA({psa_period}s)"
            for vs30 in vs30_values:
                y_position = 0
                for mag in mag_dict[tect_type]:
                    color_index = 0
                    for model in im_models[const.PSA_IM_NAME]:
                        # To match the color with global version
                        if model.endswith("NZ"):
                            color_index -= 1
                        ax[x_position, y_position].plot(
                            rrup_values.get(tect_type),
                            result_dict[tect_type][const.PSA_IM_NAME][vs30][mag][
                                psa_period
                            ][model],
                            label=model,
                            color=const.DEFAULT_LABEL_COLOR[color_index],
                            linestyle="dashed" if model.endswith("NZ") else "solid",
                        )
                        color_index += 1

                    ax[x_position, y_position].set_title(
                        f"{im_label} versus Rrup - Mw{mag}, Vs30-{vs30}"
                    )

                    ax[x_position, y_position].legend(im_models[const.PSA_IM_NAME])
                    ax[x_position, y_position].xaxis.set_label_text("Rrup [km]")
                    ax[x_position, y_position].yaxis.set_label_text("SA [g]")
                    ax[x_position, y_position].set_xscale("log")
                    ax[x_position, y_position].set_yscale("log")
                    ax[x_position, y_position].set_xlim(left=10)
                    ax[x_position, y_position].xaxis.grid(
                        True, which="both", linestyle="dotted"
                    )
                    ax[x_position, y_position].yaxis.grid(
                        True, which="both", linestyle="dotted"
                    )
                    y_position += 1
                x_position += 1

            fig.tight_layout()
            plt.savefig(
                f"{plot_directory}/{tect_type}_{im_label.replace('.', 'p')}.png"
            )
            plt.close()


def im_rrup_plot(
    mag_dict: Dict,
    vs30_values: List,
    rrup_values: Dict,
    result_dict: Dict,
    save_path: pathlib.PosixPath = None,
):
    """Plot function for a IM(no pSA) versus Rrup

    Parameters
    ----------
    mag_dict: Dict
        Dictionary with a different Mw lists for a different tectonic type
    vs30_values: List
        list of Vs30s
    rrup_values: Dict
        dictionary of Rrups np.ndarray
    result_dict: Dict
        nested dictionary that has a different computed GMMs
    save_path: pathlib.PosixPath
        Directory to save plots
    """
    root_path = (
        pathlib.Path(__file__).resolve().parent.parent
        if save_path is None
        else save_path
    )
    plot_directory = root_path / "im_rrup"
    plot_directory.mkdir(exist_ok=True, parents=True)

    plot_im_rrup(vs30_values, mag_dict, rrup_values, result_dict, plot_directory)


def psa_rrup_plot(
    mag_dict: Dict,
    vs30_values: List,
    psa_periods: List,
    rrup_values: Dict,
    result_dict: Dict,
    save_path: pathlib.PosixPath = None,
):
    """Plot function for a IM(pSA) versus Rrup

    Parameters
    ----------
    mag_dict: Dict
        Dictionary with a different Mw lists for a different tectonic type
    vs30_values: List
        list of Vs30s
    psa_periods: List
        list of Periods
    rrup_values: Dict
        dictionary of Rrups np.ndarray
    result_dict: Dict
        nested dictionary that has a different computed GMMs
    save_path: pathlib.PosixPath
        Directory to save plots
    """

    root_path = (
        pathlib.Path(__file__).resolve().parent.parent
        if save_path is None
        else save_path
    )
    plot_directory = root_path / "im_rrup"
    plot_directory.mkdir(exist_ok=True, parents=True)

    plot_psa_rrup(
        vs30_values, mag_dict, psa_periods, rrup_values, result_dict, plot_directory
    )


if __name__ == "__main__":
    mag_dict = {
        "ACTIVE_SHALLOW": [6, 7, 8],
        "SUBDUCTION_SLAB": [6, 7],
        "SUBDUCTION_INTERFACE": [7, 8, 9],
    }
    vs30_lists = [200, 400, 760]
    psa_lists = [1.0, 3.0]
    # For ACTIVE_SHALLOW and INTERFACE
    first_rrup_lists = np.linspace(1, 1000, 500)
    # For SLAB
    second_rrup_lists = np.linspace(50, 1000, 500)
    rrup_dict = {
        "ACTIVE_SHALLOW": first_rrup_lists,
        "SUBDUCTION_INTERFACE": first_rrup_lists,
        "SUBDUCTION_SLAB": second_rrup_lists,
    }
    # Update the path to the directory to save plots
    save_path = pathlib.Path("/home/tom/Documents/QuakeCoRE/verification_plots")

    start_time = time.time()

    # Set up process
    faults, result_dict = init_setup(mag_dict, vs30_lists, psa_lists)
    sites = get_sites(vs30_lists, rrup_dict)
    faults = get_faults(vs30_lists, mag_dict, faults)
    computed_gmms_dict = get_computed_gmms(
        vs30_lists, sites, mag_dict, faults, result_dict, psa_lists
    )
    setup_done = time.time()

    plot_start = time.time()

    im_rrup_plot(mag_dict, vs30_lists, rrup_dict, computed_gmms_dict, save_path)
    psa_rrup_plot(
        mag_dict, vs30_lists, psa_lists, rrup_dict, computed_gmms_dict, save_path
    )
    plot_done = time.time()
    print("START NOW!: ", start_time)
    print("SET UP IS DONE!: ", setup_done - start_time)
    print("START PLOTTING!: ", plot_start - start_time)
    print("STOP PLOTTING!: ", plot_done - start_time)
