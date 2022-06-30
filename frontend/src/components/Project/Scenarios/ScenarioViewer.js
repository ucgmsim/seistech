import React, { Fragment, useContext, useState, useEffect } from "react";

import $ from "jquery";
import Select from "react-select";
import { Tabs, Tab } from "react-bootstrap";

import { GlobalContext } from "context";
import * as CONSTANTS from "constants/Constants";
import { useAuth0 } from "components/common/ReactAuth0SPA";

import {
  LoadingSpinner,
  DownloadButton,
  GuideMessage,
  ErrorMessage,
  ScenarioPlot,
  MetadataTable,
} from "components/common";
import { getProjectScenario } from "apis/ProjectAPI";
import {
  handleErrors,
  APIQueryBuilder,
  createStationID,
  createSelectArray,
} from "utils/Utils";

import "assets/style/ScenarioViewer.css";

const ScenarioViewer = () => {
  const { isAuthenticated, getTokenSilently } = useAuth0();

  const {
    projectId,
    projectLocation,
    projectLocationCode,
    projectVS30,
    projectZ1p0,
    projectZ2p5,
    projectScenarioGetClick,
    setProjectScenarioGetClick,
    projectScenarioData,
    setProjectScenarioData,
    projectScenarioSelectedRuptures,
    projectSelectedScenarioIMComponent,
    projectSiteSelectionGetClick,
  } = useContext(GlobalContext);

  // For fetching Scenario data
  const [isLoading, setIsLoading] = useState(false);
  const [showErrorMessage, setShowErrorMessage] = useState({
    isError: false,
    errorCode: null,
  });

  // For Scenario Plots
  const [extraInfo, setExtraInfo] = useState({});

  // For Download data button
  const [downloadToken, setDownloadToken] = useState("");

  // For Select, dropdown
  const [localSelectedRP, setLocalSelectedRP] = useState(null);
  const [disaggRPOptions, setDisaggRPOptions] = useState([]);

  // Reset tabs if users click Get button from Site Selection
  useEffect(() => {
    if (projectSiteSelectionGetClick !== null) {
      setProjectScenarioGetClick(null);
      setShowErrorMessage({ isError: false, errorCode: null });
    }
  }, [projectSiteSelectionGetClick]);

  // Get Scenario data
  useEffect(() => {
    const abortController = new AbortController();
    const signal = abortController.signal;

    if (
      projectScenarioGetClick !== null &&
      projectSelectedScenarioIMComponent !== null
    ) {
      setIsLoading(true);
      setProjectScenarioData(null);
      setShowErrorMessage({ isError: false, errorCode: null });

      let token = null;
      const queryString = APIQueryBuilder({
        project_id: projectId["value"],
        station_id: createStationID(
          projectLocationCode[projectLocation],
          projectVS30,
          projectZ1p0,
          projectZ2p5
        ),
        im_component: projectSelectedScenarioIMComponent,
      });

      (async () => {
        if (isAuthenticated) token = await getTokenSilently();

        getProjectScenario(queryString, signal, token)
          .then(handleErrors)
          .then(async (response) => {
            const responseData = await response.json();
            updateScenarioData(responseData);
          })
          .catch((error) => catchError(error));
      })();
    }

    return () => {
      abortController.abort();
    };
  }, [projectScenarioGetClick]);

  const updateScenarioData = (data) => {
    setProjectScenarioData(data);
    setDownloadToken(data["download_token"]);

    const medataDataOptions = createSelectArray(Object.keys(data["metadata"]));
    setDisaggRPOptions(medataDataOptions);
    setLocalSelectedRP(medataDataOptions[0]);

    setExtraInfo({
      from: "project",
      id: projectId["value"],
      location: projectLocation,
      vs30: projectVS30,
    });

    setIsLoading(false);
  };

  const catchError = (error) => {
    if (error.name !== "AbortError") {
      setIsLoading(false);
      setShowErrorMessage({ isError: true, errorCode: error });
    }
    console.log(error);
  };

  return (
    <div className="scenario-viewer">
      <Tabs defaultActiveKey="plot" className="pivot-tabs">
        <Tab eventKey="plot" title="Plot">
          {projectScenarioGetClick === null && (
            <GuideMessage
              header={CONSTANTS.SCENARIOS}
              body={CONSTANTS.SCENARIO_VIEWER_GUIDE_MSG}
              instruction={CONSTANTS.PROJECT_SCENARIO_VIEWER_GUIDE_INSTRUCTION}
            />
          )}
          {projectScenarioGetClick !== null &&
            isLoading === true &&
            showErrorMessage.isError === false && <LoadingSpinner />}
          {isLoading === false && showErrorMessage.isError === true && (
            <ErrorMessage errorCode={showErrorMessage.errorCode} />
          )}
          {isLoading === false &&
            projectScenarioData !== null &&
            showErrorMessage.isError === false && (
              <Fragment>
                <ScenarioPlot
                  scenarioData={projectScenarioData}
                  scenarioSelectedRuptures={projectScenarioSelectedRuptures}
                  extra={extraInfo}
                />
                <DownloadButton
                  downloadURL={
                    CONSTANTS.PROJECT_API_SCENARIOS_DOWNLOAD_ENDPOINT
                  }
                  downloadToken={{
                    scenario_token: downloadToken,
                  }}
                  fileName="Scenarios.zip"
                />
              </Fragment>
            )}
        </Tab>
        <Tab eventKey="table" title="Contribution Table">
          {projectScenarioGetClick === null && (
            <GuideMessage
              header={CONSTANTS.SCENARIOS}
              body={CONSTANTS.SCENARIO_VIEWER_GUIDE_MSG}
              instruction={CONSTANTS.PROJECT_SCENARIO_VIEWER_GUIDE_INSTRUCTION}
            />
          )}
          {projectScenarioGetClick !== null &&
            isLoading === true &&
            showErrorMessage.isError === false && <LoadingSpinner />}
          {isLoading === false && showErrorMessage.isError === true && (
            <ErrorMessage errorCode={showErrorMessage.errorCode} />
          )}
          {isLoading === false &&
            projectScenarioData !== null &&
            showErrorMessage.isError === false && (
              <Fragment>
                <Select
                  value={localSelectedRP}
                  onChange={(rpOption) => setLocalSelectedRP(rpOption)}
                  options={disaggRPOptions}
                  isDisabled={disaggRPOptions.length === 0}
                  menuPlacement="auto"
                />
                <MetadataTable
                  metadata={
                    projectScenarioData["metadata"][localSelectedRP["value"]]
                  }
                  scenarioRuptures={Object.keys(
                    projectScenarioData["ensemble_scenario"]["mu_data"]
                  )}
                />
                {/* <button
                  className="btn btn-info hazard-disagg-contrib-button"
                  onClick={() => rowToggle()}
                >
                  {toggleText}
                </button> */}
              </Fragment>
            )}
        </Tab>
      </Tabs>
    </div>
  );
};

export default ScenarioViewer;
