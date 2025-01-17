import tempfile
from typing import Tuple, Dict

import flask
from flask_cors import cross_origin
from flask_caching import Cache

import gmhazard_calc as sc
import api_utils as au
from core_api import server
from core_api import constants as const
from core_api import utils


class HazardCachedData(au.api.BaseCacheData):
    """Just a wrapper for caching hazard result data"""

    def __init__(
        self,
        ensemble: sc.gm_data.Ensemble,
        site_info: sc.site.SiteInfo,
        ensemble_hazard: sc.hazard.EnsembleHazardResult,
        branches_hazard: Dict[str, sc.hazard.BranchHazardResult],
    ):
        super().__init__(ensemble, site_info)
        self.ensemble_hazard = ensemble_hazard
        self.branches_hazard = branches_hazard

    def __iter__(self):
        return iter(
            (self.ensemble, self.site_info, self.ensemble_hazard, self.branches_hazard,)
        )


@server.app.route(const.ENSEMBLE_HAZARD_ENDPOINT, methods=["GET"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@server.requires_auth
@au.api.endpoint_exception_handling(server.app)
def get_ensemble_hazard():
    """Retrieves the hazard for the ensemble, all its
     branches for the specified station (name) and NZ code

    Valid request have to contain the following
    URL parameters: ensemble_id, station, im
    Optional parameters: calc_percentiles, vs30
    """
    server.app.logger.info(f"Received request at {const.ENSEMBLE_HAZARD_ENDPOINT}")
    cache = server.cache

    (ensemble_id, station, im), optional_kwargs = au.api.get_check_keys(
        flask.request.args,
        ("ensemble_id", "station", "im"),
        (("calc_percentiles", int), ("vs30", float), ("im_component", str, "RotD50"),),
    )

    user_vs30 = optional_kwargs.get("vs30")
    calc_percentiles = optional_kwargs.get("calc_percentiles")
    calc_percentiles = False if calc_percentiles is None else bool(calc_percentiles)
    im = sc.im.IM.from_str(im, im_component=optional_kwargs.get("im_component"))

    server.app.logger.debug(
        f"Request parameters {ensemble_id}, {station}, {im}, {im.component}, {calc_percentiles}"
    )

    # Get the hazard data (either compute or from cache)
    ensemble, site_info, ensemble_hazard, branches_hazard = _get_hazard(
        ensemble_id,
        station,
        im,
        cache,
        calc_percentiles=calc_percentiles,
        user_vs30=user_vs30,
    )

    result = au.api.get_ensemble_hazard_response(
        ensemble_hazard,
        au.api.get_download_token(
            {
                "type": "ensemble_hazard",
                "ensemble_id": ensemble_id,
                "station": station,
                "user_vs30": site_info.user_vs30,
                "im": str(im),
                "im_component": str(im.component),
                "calc_percentiles": calc_percentiles,
            },
            server.DOWNLOAD_URL_SECRET_KEY,
        ),
    )

    # Adding percentiles based on flag
    if calc_percentiles:
        percentiles = {
            key: {im_value: exceedance for im_value, exceedance in value.items()}
            for key, value in ensemble_hazard.percentiles.items()
        }
        result = {**result, "percentiles": percentiles}

    return flask.jsonify(result)


@server.app.route(const.ENSEMBLE_HAZARD_DOWNLOAD_ENDPOINT, methods=["GET"])
@au.api.endpoint_exception_handling(server.app)
def download_ens_hazard():
    """Handles downloading of the hazard data

    The data is computed, saved in a temporary dictionary, zipped and
    then returned to the user
    """
    server.app.logger.info(
        f"Received request at {const.ENSEMBLE_HAZARD_DOWNLOAD_ENDPOINT}"
    )
    cache = server.cache

    (hazard_token,), optional_kwargs = au.api.get_check_keys(
        flask.request.args,
        ("hazard_token",),
        ("nzs1170p5_hazard_token", "nzta_hazard_token"),
    )
    nzs1170p5_hazard_token, nzta_hazard_token = (
        optional_kwargs.get("nzs1170p5_hazard_token"),
        optional_kwargs.get("nzta_hazard_token"),
    )

    hazard_payload = au.api.get_token_payload(
        hazard_token, server.DOWNLOAD_URL_SECRET_KEY
    )
    ensemble_id, station, user_vs30, im, calc_percentiles = (
        hazard_payload["ensemble_id"],
        hazard_payload["station"],
        hazard_payload["user_vs30"],
        sc.im.IM.from_str(
            hazard_payload["im"], im_component=hazard_payload["im_component"]
        ),
        hazard_payload["calc_percentiles"],
    )

    if nzs1170p5_hazard_token is not None:
        nzs1170p5_payload = au.api.get_token_payload(
            nzs1170p5_hazard_token, server.DOWNLOAD_URL_SECRET_KEY
        )
        nzs1170p5_im = sc.im.IM.from_str(
            nzs1170p5_payload["im"], im_component=nzs1170p5_payload["im_component"]
        )
        assert (
            ensemble_id == nzs1170p5_payload["ensemble_id"]
            and station == nzs1170p5_payload["station"]
            and im == nzs1170p5_im
        )

    if nzta_hazard_token is not None:
        nzta_payload = au.api.get_token_payload(
            nzta_hazard_token, server.DOWNLOAD_URL_SECRET_KEY
        )
        assert nzta_payload is None or (
            ensemble_id == nzta_payload["ensemble_id"],
            station == nzta_payload["station"],
        )

    # Get the hazard data (either compute or from cache)
    ensemble, site_info, ensemble_hazard, branches_hazard = _get_hazard(
        ensemble_id,
        station,
        im,
        cache,
        calc_percentiles=calc_percentiles,
        user_vs30=user_vs30,
    )

    # Get the NZS1170p5 hazard data from the cache
    nzs1170p5_hazard = None
    if nzs1170p5_hazard_token is not None:
        opt_args = {
            cur_key: cur_type(nzs1170p5_payload[cur_key])
            for cur_key, cur_type in const.NZ_CODE_OPT_ARGS
            if cur_key in nzs1170p5_payload.keys()
        }
        _, __, nzs1170p5_hazard = utils.get_nzs1170p5_hazard(
            ensemble_id, station, im, opt_args, cache, user_vs30=user_vs30
        )

    # Get the NZTA hazard data from the cache
    nzta_hazard = None
    if nzta_hazard_token is not None:
        _, __, nzta_hazard = utils.get_nzta_result(
            ensemble_id,
            station,
            sc.NZTASoilClass(nzta_payload["soil_class"]),
            cache,
            user_vs30=user_vs30,
            im_component=im.component,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_ffp = au.api.create_hazard_download_zip(
            ensemble_hazard,
            tmp_dir,
            nzs1170p5_hazard=nzs1170p5_hazard,
            nzta_hazard=nzta_hazard,
            prefix=f"{ensemble.name}",
        )

        return flask.send_file(
            zip_ffp,
            as_attachment=True,
            attachment_filename=f"{ensemble.name}_{ensemble_hazard.site.station_name}_hazard.zip",
        )


def _get_hazard(
    ensemble_id: str,
    station: str,
    im: sc.im.IM,
    cache: Cache,
    calc_percentiles: bool = False,
    user_vs30: float = None,
) -> Tuple[
    sc.gm_data.Ensemble,
    sc.site.SiteInfo,
    sc.hazard.EnsembleHazardResult,
    Dict[str, sc.hazard.BranchHazardResult],
]:
    git_version = au.api.get_repo_version()

    # Get the cached result, if there is one
    cache_key = au.api.get_cache_key(
        "hazard",
        ensemble_id=ensemble_id,
        station=station,
        vs30=str(user_vs30),
        im=str(im),
        im_component=str(im.component),
        calc_percentiles=str(calc_percentiles),
    )
    cached_data = cache.get(cache_key)

    if cached_data is None:
        server.app.logger.debug(f"No cached result for {cache_key}, computing hazard")

        server.app.logger.debug(f"Loading ensemble and retrieving site information")
        ensemble = sc.gm_data.Ensemble(ensemble_id)
        site_info = sc.site.get_site_from_name(ensemble, station, user_vs30=user_vs30)

        server.app.logger.debug(f"Computing hazard - version {git_version}")
        ensemble_hazard, branches_hazard = sc.hazard.run_full_hazard(
            ensemble, site_info, im, calc_percentiles=calc_percentiles
        )

        # Save the result
        cache.set(
            cache_key,
            HazardCachedData(ensemble, site_info, ensemble_hazard, branches_hazard),
        )
    else:
        server.app.logger.debug(f"Using cached result with key {cache_key}")
        (ensemble, site_info, ensemble_hazard, branches_hazard,) = cached_data

    return ensemble, site_info, ensemble_hazard, branches_hazard
