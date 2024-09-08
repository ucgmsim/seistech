import pickle
from pathlib import Path
from typing import Sequence, Callable, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from sklearn.neighbors import NearestNeighbors

import gmhazard_calc as gc
import sha_calc as sha
from IM_calculation.source_site_dist.src_site_dist import calc_rrup_rjb
from qcore import geo


@dataclass
class CondLnIMDistributionResult:
    """
    IM: gc.im.IM
        The IM for which this distribution is
    cond_lnIM_df: dataframe
        Conditional lnIM distribution
        for each station of interest
    obs_df: series
        Observations data series
    obs_stations: array of strings
        Array of the observation stations used
    obs_station_mask_df: dataframe
        dataframe that contains for each site
        of interest (index) specifies the relevant
        observation sites (columns) via a boolean
        mask
    combined_df: dataframe
        Contains data for both the
        sites of interest and the
        observation sites (only the
        ones with gmm parameters)
    R: dataframe
        Correlation matrix
    """

    IM: gc.im.IM
    cond_lnIM_df: pd.DataFrame
    obs_stations: np.ndarray
    obs_series: pd.Series
    obs_stations_mask_df: pd.DataFrame
    # combined_df: pd.DataFrame

    R: pd.DataFrame

    def save(self, output_ffp: Path):
        with open(output_ffp, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, data_ffp: Path):
        with open(data_ffp, "rb") as f:
            return pickle.load(f)


def get_nn_obs_site_filter_fn(
    min_n_obs_sites: int,
    n_obs_sites: int,
):
    """
    Returns the nearest neighbor observation site
    filter function

    Parameters
    ----------
    min_n_obs_sites: int
        Minimum number of observation sites
        to use
    n_obs_sites: int
        Number of observation sites to use

    Returns
    -------
    callable/None
        Function that returns the observation site
        filter for each site of interest

        None if there are not enough
        observation sites available
    """

    def obs_site_filter(
        hypo_loc: Tuple[float, float],
        station_df: pd.DataFrame,
        int_stations: np.ndarray,
        obs_stations: np.ndarray,
        distance_matrix: pd.DataFrame,
    ):
        """
        Returns the observation site filter for
        each site of interest using nearest neighbor.
        """
        # Not enough observation sites
        if obs_stations.size < min_n_obs_sites:
            return None

        # Less observation sites than required but more than minimum
        # i.e. use all available
        if obs_stations.size <= n_obs_sites:
            obs_station_mask_df = pd.DataFrame(
                index=int_stations, columns=obs_stations, data=True
            )
        # More than required use nearest neighbor to select
        else:
            # Get observation sites such that minimum number of observations sites is satisfied
            neigh = NearestNeighbors(
                n_neighbors=n_obs_sites + 1, radius=150, metric="precomputed", n_jobs=1
            )
            neigh.fit(distance_matrix.loc[obs_stations, obs_stations])
            n_neigh_ind = neigh.kneighbors(
                distance_matrix.loc[int_stations, obs_stations], return_distance=False
            )

            # Convert to a mask
            n_neigh_mask = np.zeros((int_stations.size, obs_stations.size), dtype=bool)
            np.put_along_axis(n_neigh_mask, n_neigh_ind, True, axis=1)

            # Combine & Create dataframe
            obs_station_mask_df = pd.DataFrame(
                index=int_stations, columns=obs_stations, data=n_neigh_mask
            )

        # Don't include stations of interest
        int_obs_sites = int_stations[np.isin(int_stations, obs_stations)]
        for cur_station in int_obs_sites:
            obs_station_mask_df.loc[cur_station, cur_station] = False

        return obs_station_mask_df

    return obs_site_filter


def get_rmin_obs_site_filter_fn(
    min_n_obs_sites: int = 20,
):
    def obs_site_filter(
        hypo_loc: Tuple[float, float],
        station_df: pd.DataFrame,
        int_stations: np.ndarray,
        obs_stations: np.ndarray,
        distance_matrix: pd.DataFrame,
    ):
        """
        Computes the mask that specifies the relevant
        observation sites for each site of interest

        Parameters
        ----------
        hypo_loc: pair of floats
            Hypocenter location
        station_df: dataframe
            Dataframe that contains the station
            locations
            Must have columns: [lon, lat]
        int_stations: array of strings
            Name of the stations of interest
        obs_stations: array of strings
            Name of the observation stations
        distance_matrix: dataframe
            Distance matrix between all sites

        Returns
        -------
        dataframe
            Dataframe that contains mask of
            observation sites to use
            for each site of interest

            Index: sites of interest
            Columns: observation sites
        """
        # Compute R_min for each site of interest
        src_site_dist = pd.Series(
            data=geo.get_distances(
                station_df.loc[int_stations, ["lon", "lat"]].values,
                hypo_loc[0],
                hypo_loc[1],
            ),
            index=int_stations,
        )
        r_min = src_site_dist * 1.5

        # Sanity check
        assert np.all(r_min.index == distance_matrix.loc[int_stations].index)

        # Get observation sites that are within r_min
        r_min_mask = (
            distance_matrix.loc[int_stations, obs_stations].values
            < r_min.values[:, np.newaxis]
        )

        # Get observation sites such that minimum number of observations sites is satisfied
        neigh = NearestNeighbors(
            n_neighbors=min_n_obs_sites + 1, radius=150, metric="precomputed", n_jobs=1
        )
        neigh.fit(distance_matrix.loc[obs_stations, obs_stations])
        n_neigh_ind = neigh.kneighbors(
            distance_matrix.loc[int_stations, obs_stations], return_distance=False
        )

        # Convert to a mask
        n_neigh_mask = np.zeros_like(r_min_mask)
        np.put_along_axis(n_neigh_mask, n_neigh_ind, True, axis=1)

        # Combine & Create dataframe
        obs_station_mask_df = pd.DataFrame(
            index=int_stations, columns=obs_stations, data=r_min_mask | n_neigh_mask
        )

        # Don't include stations of interest
        int_obs_sites = int_stations[np.isin(int_stations, obs_stations)]
        for cur_station in int_obs_sites:
            obs_station_mask_df.loc[cur_station, cur_station] = False

        return obs_station_mask_df

    return obs_site_filter


def get_max_dist_obs_site_filter_fn(
    max_obs_dist: float,
    min_n_obs: int
):
    def obs_site_filter(
        hypo_loc: Tuple[float, float],
        station_df: pd.DataFrame,
        int_stations: np.ndarray,
        obs_stations: np.ndarray,
        distance_matrix: pd.DataFrame,
    ):
        """
        Computes the mask that specifies the relevant
        observation sites for each site of interest.

        Selects all observation sites within `max_dist`
        of the site of interest.
        If there is at least one observation site within
        `max_dist` then addtionaly records are selected until
        `min_n_obs` is satisfied using nearest neighbor

        Parameters
        ----------
        hypo_loc: pair of floats
            Hypocenter location
        station_df: dataframe
            Dataframe that contains the station
            locations
            Must have columns: [lon, lat]
        int_stations: array of strings
            Name of the stations of interest
        obs_stations: array of strings
            Name of the observation stations
        distance_matrix: dataframe
            Distance matrix between all sites

        Returns
        -------
        dataframe
            Dataframe that contains mask of
            observation sites to use
            for each site of interest

            Index: sites of interest
            Columns: observation sites
        """
        obs_station_mask_df = pd.DataFrame(
            index=int_stations,
            columns=obs_stations,
            data=False,
        )

        # Get observation sites such that minimum number of observations sites is satisfied
        neigh = NearestNeighbors(
            n_neighbors=min_n_obs + 1, radius=150, metric="precomputed", n_jobs=1
        )
        neigh.fit(distance_matrix.loc[obs_stations, obs_stations])
        n_neigh_ind = neigh.kneighbors(
            distance_matrix.loc[int_stations, obs_stations], return_distance=False
        )

        # Set `min_n_obs` observation sites to True
        np.put_along_axis(obs_station_mask_df.values, n_neigh_ind, True, axis=1)

        # Get observation sites within `max_obs_dist`
        max_dist_mask = distance_matrix.loc[int_stations, obs_stations] < max_obs_dist

        # If there are no observation sites within `max_obs_dist`
        # then set all observation sites to False
        obs_station_mask_df.values[~np.any(max_dist_mask, axis=1), :] = False

        # Enable all observation sites within `max_obs_dist`
        obs_station_mask_df |= max_dist_mask

        # Don't include stations of interest
        int_obs_sites = int_stations[np.isin(int_stations, obs_stations)]
        for cur_station in int_obs_sites:
            obs_station_mask_df.loc[cur_station, cur_station] = False

        return obs_station_mask_df

    return obs_site_filter


def compute_cond_lnIM(
    IM: gc.im.IM,
    int_stations: np.ndarray,
    stations_df: pd.DataFrame,
    gmm_params_df: pd.DataFrame,
    obs_series: pd.Series,
    hypo_loc: Tuple[float, float],
    obs_site_filter_fn: Callable[[pd.DataFrame], List[str]],
    R: pd.DataFrame = None,
    allow_obs_sites: bool = False,
    verbose: bool = True,
) -> CondLnIMDistributionResult:
    """
    Computes the conditional lnIM distribution
    for each station of interest

    Parameters
    ----------
    IM: IM
        IM for which to compute
        the conditional IM distribution
    int_stations: sequence of strings
        The stations of interest
    stations_df: dataframe
    gmm_params_df: dataframe
        Dataframe with empirical GMMs parameters

        Expects columns names
        [{IM}_mean, {IM}_std_Total, {IM}_std_Inter, {IM}_std_Intra]
    obs_series: series
        Observations IM values for each station
    hypo_loc: pair of floats
        The lon and lat value of the hypocentre
    R: dataframe
        Within-event site correlation matrix, optional

        If not provided then this is computed using the
        Loth & Baker model (2013)
    obs_site_filter_fn: callable
        Function that performs filtering on the distance
        between the site of interest and the available
        observation sites

        Use this to prevent global bias (wrt. to the empirical
        GMM) affecting the between-event term calculation

        Takes following arguments
            hypo_loc: Tuple[float, float],
            station_df: pd.DataFrame,
            int_stations: np.ndarray,
            obs_stations: np.ndarray,
            distance_matrix: pd.DataFrame,
        and returns a dataframe with
            index=sites of interest
            columns=observation sites
            data=boolean mask
    allow_obs_sites: bool
        If True then any observation sites in the
        sites of interests will also be computed
        as if that observation does not exist
    verbose: bool

    Returns
    -------
    CondLnIMDistributionResult
    """
    obs_stations = obs_series.index.values.astype(str)

    # Check that GMM data exists for all observed stations
    # Otherwise drop those stations
    mask = np.isin(obs_stations, gmm_params_df.index.values)
    if np.count_nonzero(~mask) > 0:
        if verbose:
            print(
                f"Missing GMM parameters for (observation) stations:\n"
                f"{obs_stations[~mask]}\n\tDropping these stations"
            )
        obs_stations = obs_stations[mask]
    obs_series = obs_series.loc[obs_stations]

    # Check that GMM data exists for all stations of interest
    # Otherwise drop them
    mask = np.isin(int_stations, gmm_params_df.index.values)
    if np.count_nonzero(~mask) > 0:
        if verbose:
            print(
                f"Missing GMM parameters for sites of interest:\n"
                f"{int_stations[~mask]}\n\tDropping these stations"
            )
        int_stations = int_stations[mask]

    # Drop any sites of interest for which observations exists
    mask = np.isin(int_stations, obs_stations)
    if np.count_nonzero(mask) > 0 and not allow_obs_sites:
        if verbose:
            print(
                f"Observations exist for the following sites of interest:\n"
                f"{int_stations[mask]}\n\tDropping these stations"
            )
        int_stations = int_stations[~mask]

    if verbose:
        print(
            f"Computing results for {int_stations.size} stations of interest, "
            f"with {obs_stations.size} observation stations available"
        )

    # Relevant stations (Observation sites & Sites of interest)
    rel_stations = np.union1d(int_stations, obs_stations)
    gmm_params_df = gmm_params_df.loc[rel_stations]

    if verbose:
        print("Computing distance matrix")
    dist_matrix = calculate_distance_matrix(rel_stations, stations_df)

    if R is None:
        if verbose:
            print("Computing within-event site correlation matrix")
        R = get_corr_matrix(rel_stations, dist_matrix, IM)
    elif verbose:
        print("Using provided within-event site correlation matrix")

    # Get the observation sites to use for each site of interest
    if (
        obs_station_mask_df := obs_site_filter_fn(
            hypo_loc, stations_df, int_stations, obs_stations, dist_matrix
        )
    ) is None:
        if verbose:
            print("Not enough observation sites available, skipping")
        return None
    assert np.all(obs_station_mask_df.columns.values.astype(str) == obs_stations)
    assert np.all(obs_station_mask_df.index.values.astype(str) == int_stations)

    # Compute the conditional MVN for each site of interest
    cond_df = pd.DataFrame(
        data=np.full((int_stations.shape[0], 2), np.nan),
        index=int_stations,
        columns=["mu", "sigma"],
    )
    for cur_station in int_stations:
        cur_mask = obs_station_mask_df.loc[cur_station].values
        cur_rel_sites = np.concatenate(([cur_station], obs_stations[cur_mask]))
        cur_cond_mu, cur_cond_sigma = sha.compute_cond_lnIM_dist(
            cur_station,
            gmm_params_df.loc[cur_rel_sites],
            obs_series.loc[obs_stations[cur_mask]],
            R.loc[cur_rel_sites, cur_rel_sites],
        )

        cond_df.loc[cur_station, ["mu", "sigma"]] = cur_cond_mu, cur_cond_sigma

    # # Combine into single data frame
    # combined_df = pd.concat((cond_df, obs_series.to_frame("mu")), axis=0)
    # combined_df["observation"] = False
    # combined_df.loc[obs_stations, "observation"] = True
    # combined_df.loc[obs_stations, "sigma"] = 0.0
    #
    # # Compute median
    # combined_df["median"] = np.exp(combined_df.mu)

    return CondLnIMDistributionResult(
        IM, cond_df, obs_stations, obs_series, obs_station_mask_df, R
    )


def calculate_distance_matrix(stations: Sequence[str], locations_df: pd.DataFrame, site_lon_col: str = "lon", site_lat_col: str = "lat"):
    """
    Given a set of stations and their locations (in lat, lon format),
    calculate the matrix containing
    the pairwise distance

    Parameters
    ----------
    stations: Sequence[str]
        List of the station names
    locations_df: pd.DataFrame
        Locations of each of the stations (in lat, lon)
    """
    distance_matrix = -1 * np.ones((len(stations), len(stations)))
    for i, station in enumerate(stations):
        cur_dist, _ = calc_rrup_rjb(
            np.asarray(
                [[locations_df.loc[station, site_lon_col], locations_df.loc[station, site_lat_col], 0]]
            ),
            np.stack(
                (
                    locations_df.loc[stations, site_lon_col],
                    locations_df.loc[stations, site_lat_col],
                    np.zeros(len(stations)),
                ),
                axis=1,
            ),
        )
        distance_matrix[i, :] = cur_dist
    return pd.DataFrame(index=stations, data=distance_matrix, columns=stations)


def get_corr_matrix(
    stations: Sequence[str], distance_matrix: pd.DataFrame, selected_im: gc.im.IM
):
    """
    Computes the correlation matrix for the specified stations
    using the Loth & Baker (2013) model.

    Parameters
    ----------
    stations: Sequence[str]
        List of the station names
    distance_matrix: pd.DataFrame
        Distance values between all site locations
    selected_im: IM
        IM to get correlations from the model
    """
    R = np.eye(len(stations))
    for i, station in enumerate(stations):
        correlation = sha.loth_baker_corr_model.get_correlations(
            str(selected_im),
            str(selected_im),
            distance_matrix.loc[stations, station].values,
        )
        assert np.all(
            correlation >= 0.0
        ), f"Correlation should be positive or 0. Error with {station} at index {i}"
        R[i, :] = R[:, i] = correlation

    # Make the diagonal values exactly 1.0
    assert np.all(np.isclose(np.diag(R), 1.0, rtol=1e-2))
    np.fill_diagonal(R, 1.0)

    return pd.DataFrame(index=stations, columns=stations, data=R)


def gen_spatial_im_rels(
    N: int,
    R: pd.DataFrame,
    emp_df: pd.DataFrame,
):
    """
    Given a number of stations with their empirical values,
     create uncorrelated random values and then
    correlate them using the Cholesky decomposition
     of the correlation matrix (R).

    Note: The generated IM values are only correlated
        spatially, inter-correlation between the IMs
        is not considered in this implementation!

    Parameters
    ----------
    N: int
        Number of realisations to generate
    R: pd.Dataframe
        Within-event correlation matrix
    emp_df: pd.Dataframe
        Empirical results with columns
         mu, between_event_sigma and within_event_sigma
    """
    mean_lnIM, between_event_std, within_event_std = (
        emp_df["mu"],
        emp_df["between_event_sigma"],
        emp_df["within_event_sigma"],
    )

    # Cholesky decomposition of the correlation matrix
    # Make positive definite if it isn't
    try:
        L = cholesky(R, lower=True)
    except np.linalg.LinAlgError:
        pd_R = sha.nearest_pd(R)
        L = cholesky(pd_R, lower=True)

    # Calculate random between event residual value
    #  per realisation and multiply by between event sigma
    between_event = (
        np.random.normal(0.0, 1.0, size=N)[:, np.newaxis]
        * between_event_std.values[np.newaxis, :]
    )

    # Calculate random within event residual value
    #  per site and per realisation
    # Then matrix multiply against the result of
    #  the Cholesky decomposition for each realisation
    within_event = np.matmul(
        L, np.random.normal(0.0, within_event_std, size=(N, len(within_event_std))).T
    ).T

    # Combine the between and within event values with
    #  the mean IM values broadcast across realisations
    im_values = mean_lnIM.values[None, :] + between_event + within_event

    return im_values, between_event, within_event


def gen_im_rels(
    gm_params: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    ims: Sequence[str],
    n_rels: int,
    corr_fn: Callable[
        [str, str, np.ndarray], np.ndarray
    ] = sha.models.loth_baker_corr_model.get_correlations,
):
    """
    Generates cross-correlated IM realisations for the given
    sites and ground motion parameters

    Parameters
    ----------
    gm_params: dataframe
        GM parameters, must have the following columns
        for each IM {IM}_mean, {IM}_std_Inter, {IM}_std_Intra
    dist_matrix: dataframe
    ims: sequence of strings
    n_rels: int
    corr_fn: callable
        The correlation function to use, must take the
        following arguments:
            IM_1: str
            IM_2: str
                The two IMs for which to compute the correlation
            dist: array of floats
                The distance values between the sites
        and return the correlation values between the two IMs
        for each distance value

    Returns
    -------
    im_values: array of floats
        The generated IM realisations, shape:
        [n_sites, n_ims, n_rels]
    """
    n_ims, n_sites = len(ims), dist_matrix.shape[0]
    sites = dist_matrix.index.values

    # Create the correlation matrix
    # The correlation matrix is made up of smaller
    # block matrices (n_sites x n_sites) representing the correlation
    # between the different IMs
    # I.e.
    # [R^1,1, R^1,2, ..., R^1,m],
    # [R^2,1, R^2,2, ..., R^2,m],
    # [...                     ],
    # [R^m,1, R^m,2, ..., R^m,m],
    # where R^i,j is the correlation between IM_i and IM_j
    # for all sites considered
    R = np.full((n_sites * n_ims, n_sites * n_ims), fill_value=np.nan)
    for i, im_i in enumerate(ims):
        for j, im_j in enumerate(ims):
            R[i * n_sites : (i + 1) * n_sites, j * n_sites : (j + 1) * n_sites] = (
                corr_fn(im_i, im_j, dist_matrix.values)
            )

    # Cholesky decomposition of the correlation matrix
    # Make positive definite if it isn't
    try:
        L = cholesky(R, lower=True)
    except np.linalg.LinAlgError:
        pd_R = sha.nearest_pd(R)
        L = cholesky(pd_R, lower=True)

    # Generate the between event-values
    mu = np.full((n_sites * n_ims, n_rels), fill_value=np.nan)
    between_event_residual = np.full((n_sites * n_ims, n_rels), fill_value=np.nan)
    u = np.full((n_sites * n_ims, n_rels), fill_value=np.nan)
    for i, im in enumerate(ims):
        between_event_residual[i * n_sites : (i + 1) * n_sites, :] = (
            np.tile(np.random.normal(0.0, 1.0, size=n_rels), (n_sites, 1))
            * gm_params.loc[sites, f"{im}_std_Inter"].values[:, np.newaxis]
        )

        u[i * n_sites : (i + 1) * n_sites] = (
            np.random.normal(0.0, 1.0, size=n_sites * n_rels).reshape((n_sites, n_rels))
            * gm_params.loc[sites, f"{im}_std_Intra"].values[:, np.newaxis]
        )

        mu[i * n_sites : (i + 1) * n_sites, :] = np.tile(
            gm_params.loc[sites, f"{im}_mean"].values[:, np.newaxis], (1, n_rels)
        )

    # Generate the within-event values
    within_event_residual = np.einsum("ij,jk->ik", L, u)

    im_values = mu + between_event_residual + within_event_residual

    # Convert back to [n_sites, n_ims, n_rels]
    im_values = im_values.reshape((n_sites, n_ims, n_rels), order="F")
    between_event_residual = between_event_residual.reshape(
        (n_sites, n_ims, n_rels), order="F"
    )
    within_event_residual = within_event_residual.reshape(
        (n_sites, n_ims, n_rels), order="F"
    )

    return im_values, between_event_residual, within_event_residual
