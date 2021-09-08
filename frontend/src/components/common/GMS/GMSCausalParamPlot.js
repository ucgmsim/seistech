import React from "react";

import Plot from "react-plotly.js";

import { PLOT_MARGIN, PLOT_CONFIG, GMS_LABELS } from "constants/Constants";
import { range, sortDuplicateXRange, sortDuplicateYRange } from "utils/Utils";

import "assets/style/GMSPlot.css";

// Mainly for Vs30 and SF as Mag and Rrup will use DisaggDistributionPlot
const GMSCausalParamPlot = ({ gmsData, metadata, causalParamBounds }) => {
  /*
    Create a new array to avoid affecting the original object from any modification
    then sort metadata & duplicate every element
  */
  const xRange = sortDuplicateXRange([
    ...gmsData["selected_gms_metadata"][metadata],
  ]);

  // Create an array between 0 and 1 by step(third param), refer to the function for more detail
  const rangeY = range(
    0,
    1,
    1 / gmsData["selected_gms_metadata"][metadata].length
  );

  // Duplicate the elements except the first and the last element
  const newRangeY = sortDuplicateYRange(rangeY);

  const scattersArray = [
    {
      x: xRange,
      y: newRangeY,
      mode: "lines+markers",
      name:
        GMS_LABELS[metadata] === "Vs30"
          ? "Selected GMs"
          : `${GMS_LABELS[metadata]}`,
      line: { shape: "hv", color: "black" },
      type: "scatter",
      showlegend: true,
    },
  ];

  // Y coordiates for bounds
  const boundsRangeY = [0, 1];

  // Add bounds (Min / Max)
  if (causalParamBounds[metadata]) {
    scattersArray.push(
      {
        x: [
          causalParamBounds[metadata]["min"],
          causalParamBounds[metadata]["min"],
        ],
        y: boundsRangeY,
        legendgroup: metadata,
        name: "Lower and upper bound limits",
        mode: "lines",
        line: { color: "red", dash: "dot" },
        type: "scatter",
      },
      {
        x: [
          causalParamBounds[metadata]["max"],
          causalParamBounds[metadata]["max"],
        ],
        y: boundsRangeY,
        legendgroup: metadata,
        name: "Lower and upper bound limits",
        mode: "lines",
        line: { color: "red", dash: "dot" },
        type: "scatter",
        showlegend: false,
      }
    );
  }
  // Add solid line for vs30 on top of Min/Max
  if (metadata === "vs30") {
    scattersArray.push({
      x: [
        causalParamBounds[metadata]["vs30"],
        causalParamBounds[metadata]["vs30"],
      ],
      y: boundsRangeY,
      legendgroup: metadata,
      name: `Site-Specific V${"s30".sub()}`,
      mode: "lines",
      line: { color: "red" },
      type: "scatter",
    });
  } else if (metadata === "sf") {
    // Add a solid red line at x=1 as a reference point
    scattersArray.push({
      x: [1, 1],
      y: boundsRangeY,
      name: "Reference Point",
      mode: "lines",
      line: { color: "red" },
      type: "scatter",
    });
  }

  // Setting xRange with a same padding on both sides
  let minXCoord = causalParamBounds[metadata]
    ? Math.min(...xRange, causalParamBounds[metadata]["min"])
    : Math.min(...xRange, 1);
  let maxXCoord = causalParamBounds[metadata]
    ? Math.max(...xRange, causalParamBounds[metadata]["max"])
    : Math.max(...xRange, 1);

  const xAxisGap = maxXCoord * 0.1;
  const xAxisRange = [minXCoord - xAxisGap, maxXCoord + xAxisGap];

  return (
    <Plot
      className={"fourth-plot"}
      data={scattersArray}
      layout={{
        xaxis: {
          title: { text: `${GMS_LABELS[metadata]} distribution` },
          range: xAxisRange,
          autorange: false,
        },
        yaxis: {
          title: { text: "Cumulative Probability, CDF" },
          autorange: true,
        },
        autosize: true,
        margin: PLOT_MARGIN,
      }}
      useResizeHandler={true}
      config={PLOT_CONFIG}
    />
  );
};

export default GMSCausalParamPlot;