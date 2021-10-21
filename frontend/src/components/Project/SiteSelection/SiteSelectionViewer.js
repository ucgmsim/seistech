import React, { Fragment, useContext, useEffect, useState } from "react";

import { Tab, Nav } from "react-bootstrap";

import { GlobalContext } from "context";
import { useAuth0 } from "components/common/ReactAuth0SPA";
import * as CONSTANTS from "constants/Constants";

import {
  LoadingSpinner,
  GuideMessage,
  ErrorMessage,
  ImageMap,
} from "components/common";
import { getProjectMaps } from "apis/ProjectAPI";
import { getPublicProjectMaps } from "apis/PublicProjectAPI";
import { handleErrors, APIQueryBuilder, createStationID } from "utils/Utils";

const SiteSelectionViewer = () => {
  const { isAuthenticated, getTokenSilently } = useAuth0();

  const {
    projectId,
    projectLocation,
    projectVS30,
    projectZ1p0,
    projectZ2p5,
    projectLocationCode,
    projectSiteSelectionGetClick,
    setProjectSiteSelectionGetClick,
  } = useContext(GlobalContext);

  const [showSpinner, setShowSpinner] = useState(false);
  const [showImages, setShowImages] = useState(false);
  const [showErrorMessage, setShowErrorMessage] = useState({
    isError: false,
    errorCode: null,
  });

  const [regionalMap, setRegionalMap] = useState(null);
  const [vs30Map, setVS30Map] = useState(null);

  /*
    Reset when it first renders
    E.g. change tab between Projects to Hazard Analyis
  */
  useEffect(() => {
    setShowImages(false);
    setShowSpinner(false);
    setProjectSiteSelectionGetClick(null);
  }, []);

  // Get Context/Vs30 maps
  useEffect(() => {
    const abortController = new AbortController();
    const signal = abortController.signal;

    if (projectSiteSelectionGetClick !== null) {
      let queryString = APIQueryBuilder({
        project_id: projectId["value"],
        station_id: createStationID(
          projectLocationCode[projectLocation],
          projectVS30,
          projectZ1p0,
          projectZ2p5
        ),
      });
      setShowImages(false);
      setShowSpinner(true);
      setShowErrorMessage({ isError: false, errorCode: null });

      if (isAuthenticated) {
        (async () => {
          const token = await getTokenSilently();

          getProjectMaps(signal, token, queryString)
            .then(handleErrors)
            .then(async (response) => {
              const responseData = await response.json();
              updateMap(responseData);
            })
            .catch((error) => {
              if (error.name !== "AbortError") {
                setShowSpinner(false);
                setShowErrorMessage({ isError: true, errorCode: error });
              }
              console.log(error);
            });
        })();
      } else {
        getPublicProjectMaps(signal, queryString)
          .then(handleErrors)
          .then(async (response) => {
            const responseData = await response.json();
            updateMap(responseData);
          })
          .catch((error) => {
            if (error.name !== "AbortError") {
              setShowSpinner(false);
              setShowErrorMessage({ isError: true, errorCode: error });
            }
            console.log(error);
          });
      }
    }

    return () => {
      abortController.abort();
    };
  }, [projectSiteSelectionGetClick]);

  const updateMap = (mapData) => {
    setRegionalMap(mapData["context_plot"]);
    setVS30Map(mapData["vs30_plot"]);
    setShowSpinner(false);
    setShowImages(true);
  };

  return (
    <Fragment>
      <Tab.Container defaultActiveKey="regional">
        <Nav variant="tabs">
          <Nav.Item>
            <Nav.Link eventKey="regional">Regional</Nav.Link>
          </Nav.Item>
          <Nav.Item>
            <Nav.Link eventKey="vs30">
              V<sub>S30</sub>
            </Nav.Link>
          </Nav.Item>
        </Nav>
        <Tab.Content>
          <Tab.Pane eventKey="regional">
            {projectSiteSelectionGetClick === null && (
              <GuideMessage
                header={CONSTANTS.SITE_SELECTION_REGIONAL_TITLE}
                body={CONSTANTS.PROJECT_SITE_SELECTION_GUIDE_MSG}
                instruction={CONSTANTS.PROJECT_SITE_SELECTION_INSTRUCTION}
              />
            )}
            {showSpinner === true &&
              projectSiteSelectionGetClick !== null &&
              showErrorMessage.isError === false && <LoadingSpinner />}

            {projectSiteSelectionGetClick !== null &&
              showSpinner === false &&
              showErrorMessage.isError === true && (
                <ErrorMessage errorCode={showErrorMessage.errorCode} />
              )}

            {showSpinner === false &&
              showImages === true &&
              regionalMap !== null &&
              showErrorMessage.isError === false && (
                <ImageMap
                  header={
                    "Looking at a map with the source locations and historical events from Geonet in the 2003-present period."
                  }
                  src={regionalMap}
                  alt={"Regional Map"}
                />
              )}
          </Tab.Pane>
          <Tab.Pane eventKey="vs30">
            {projectSiteSelectionGetClick === null && (
              <GuideMessage
                header={CONSTANTS.SITE_SELECTION_VS30_TITLE}
                body={CONSTANTS.PROJECT_SITE_SELECTION_GUIDE_MSG}
                instruction={CONSTANTS.PROJECT_SITE_SELECTION_INSTRUCTION}
              />
            )}
            {showSpinner === true &&
              projectSiteSelectionGetClick !== null &&
              showErrorMessage.isError === false && <LoadingSpinner />}

            {projectSiteSelectionGetClick !== null &&
              showSpinner === false &&
              showErrorMessage.isError === true && (
                <ErrorMessage errorCode={showErrorMessage.errorCode} />
              )}

            {showSpinner === false &&
              showImages === true &&
              regionalMap !== null &&
              showErrorMessage.isError === false && (
                <ImageMap
                  header={
                    "Looking at the Foster et al. model predictions in the vicinity of the site."
                  }
                  src={vs30Map}
                  alt={"Vs30 Map"}
                />
              )}
          </Tab.Pane>
        </Tab.Content>
      </Tab.Container>
    </Fragment>
  );
};
export default SiteSelectionViewer;
