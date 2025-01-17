import tempfile

import flask
from flask_cors import cross_origin

import api_utils as au
import gmhazard_calc as sc
import gmhazard_utils as su
from project_api import server
from project_api import constants as const
from project_api import utils


@server.app.route(const.PROJECT_HAZARD_ENDPOINT, methods=["GET"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@server.requires_auth
@au.api.endpoint_exception_handling(server.app)
def get_ensemble_hazard():
    server.app.logger.info(f"Received request at {const.PROJECT_HAZARD_ENDPOINT}")

    _, version_str = su.utils.get_package_version(const.PACKAGE_NAME)
    server.app.logger.debug(f"API - version {version_str}")

    (project_id, station_id, im), optional_kwargs = au.api.get_check_keys(
        flask.request.args,
        ("project_id", "station_id", ("im", sc.im.IM.from_str)),
        (("im_component", str),),
    )
    im.component = (
        sc.im.IMComponent.RotD50
        if optional_kwargs.get("im_component") is None
        else sc.im.IMComponent[optional_kwargs.get("im_component")]
    )
    server.app.logger.debug(f"Request parameters {project_id}, {station_id}, {im}")

    # Load the data
    ensemble_hazard, nzs1170p5_hazard, nzta_hazard = utils.load_hazard_data(
        server.BASE_PROJECTS_DIR
        / version_str
        / project_id
        / "results"
        / station_id
        / str(im.component),
        im,
    )

    result = au.api.get_ensemble_hazard_response(
        ensemble_hazard,
        au.api.get_download_token(
            {
                "project_id": project_id,
                "station_id": ensemble_hazard.site.station_name,
                "im": str(ensemble_hazard.im),
                "im_component": str(ensemble_hazard.im.component),
            },
            server.DOWNLOAD_URL_SECRET_KEY,
        ),
    )

    if ensemble_hazard.percentiles is not None:
        result = {
            **result,
            "percentiles": {
                key: {
                    im_value: exceedance for im_value, exceedance in value.items()
                }
                for key, value in ensemble_hazard.percentiles.items()
            },
        }

    if nzs1170p5_hazard is not None:
        result = {**result, "nzs1170p5_hazard": nzs1170p5_hazard.to_dict()}
    if nzta_hazard is not None:
        result = {**result, "nzta_hazard": nzta_hazard.to_dict(nan_to_string=True)}
    return flask.jsonify(result)


@server.app.route(const.PROJECT_HAZARD_DOWNLOAD_ENDPOINT, methods=["GET"])
@au.api.endpoint_exception_handling(server.app)
def download_ens_hazard():
    """Handles downloading of the hazard data,
    specified by the given token"""
    server.app.logger.info(
        f"Received request at {const.PROJECT_HAZARD_DOWNLOAD_ENDPOINT}"
    )

    _, version_str = su.utils.get_package_version(const.PACKAGE_NAME)
    server.app.logger.debug(f"API - version {version_str}")

    (token,), _ = au.api.get_check_keys(flask.request.args, ("hazard_token",))

    payload = au.api.get_token_payload(token, server.DOWNLOAD_URL_SECRET_KEY)

    project_id, station_id, im, im_component = (
        payload["project_id"],
        payload["station_id"],
        sc.im.IM.from_str(payload["im"]),
        payload["im_component"],
    )

    server.app.logger.debug(
        f"Token parameters {project_id}, {station_id}, {im}, {im_component}"
    )

    # Load the data
    ensemble_hazard, nzs1170p5_hazard, nzta_hazard = utils.load_hazard_data(
        server.BASE_PROJECTS_DIR
        / version_str
        / project_id
        / "results"
        / station_id
        / im_component,
        im,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_ffp = au.api.create_hazard_download_zip(
            ensemble_hazard,
            tmp_dir,
            nzs1170p5_hazard=nzs1170p5_hazard,
            nzta_hazard=nzta_hazard,
        )

        return flask.send_file(
            zip_ffp,
            as_attachment=True,
            attachment_filename=f"{ensemble_hazard.ensemble.name}_"
            f"{ensemble_hazard.site.station_name}_hazard.zip",
        )
