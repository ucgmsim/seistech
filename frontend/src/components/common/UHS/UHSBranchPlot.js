import React from "react";

import Plot from "react-plotly.js";

import { getPlotData } from "utils/Utils.js";
import { PLOT_MARGIN, PLOT_CONFIG } from "constants/Constants";
import { ErrorMessage } from "components/common";

import "assets/style/UHSPlot.css";

const UHSBranchPlot = ({
  from,
  uhsData,
  uhsBranchData,
  nzs1170p5Data,
  rp,
  showNZS1170p5 = true,
  extra,
}) => {
  if (uhsData !== null && !uhsData.hasOwnProperty("error")) {
    const createLegendLabel = (isNZCode) => {
      return isNZCode === true
        ? "NZS1170.5 [RP = " + rp + "]"
        : "Site-specific [RP = " + rp + "]";
    };

    // Creating the scatter objects
    const scatterObjs = [];

    // UHS Branches scatter objs
    let dataCounter = 0;
    if (uhsBranchData !== null) {
      for (let curData of Object.values(uhsBranchData)) {
        // Skip any plots if it contains nan
        if (!curData.sa_values.includes("nan")) {
          scatterObjs.push({
            x: curData.period_values,
            y: curData.sa_values,
            type: "scatter",
            mode: "lines",
            line: { color: "grey", width: 0.8 },
            name: "Branches",
            legendgroup: "branches",
            showlegend: dataCounter === 0 ? true : false,
            hoverinfo: "none",
            hovertemplate:
              `<b>${curData.branch_name}</b><br><br>` +
              "%{xaxis.title.text}: %{x}<br>" +
              "%{yaxis.title.text}: %{y}<extra></extra>",
          });
          dataCounter += 1;
        }
      }
    }

    // Create NZS1170p5 Code UHS scatter objs
    // If object does not contain the value of NaN, we plot
    if (!Object.values(nzs1170p5Data).includes("nan")) {
      let nzs1170p5PlotData = getPlotData(nzs1170p5Data);
      // Convert the Annual exdance reate to Return period in a string format
      scatterObjs.push({
        x: nzs1170p5PlotData.index,
        y: nzs1170p5PlotData.values,
        type: "scatter",
        mode: "lines",
        line: { color: "black" },
        name: createLegendLabel(true),
        visible: showNZS1170p5,
        legendgroup: "NZS1170.5",
        showlegend: true,
        hoverinfo: "none",
        hovertemplate:
          `<b>NZS1170.5 [RP ${rp}]</b><br><br>` +
          "%{xaxis.title.text}: %{x}<br>" +
          "%{yaxis.title.text}: %{y}<extra></extra>",
      });
    }

    // UHS scatter objs
    if (!uhsData.sa_values.includes("nan")) {
      scatterObjs.push({
        x: uhsData.period_values,
        y: uhsData.sa_values,
        type: "scatter",
        mode: "lines",
        line: { color: "blue" },
        name: createLegendLabel(false),
        legendgroup: "site-specific",
        showlegend: true,
        hoverinfo: "none",
        hovertemplate:
          `<b>Site-specific [RP ${rp}]</b><br><br>` +
          "%{xaxis.title.text}: %{x}<br>" +
          "%{yaxis.title.text}: %{y}<extra></extra>",
      });
    }

    // For Percentiles
    if (uhsData.percentiles) {
      const percentile16 = getPlotData(uhsData.percentiles["16th"]);
      const percentile84 = getPlotData(uhsData.percentiles["84th"]);

      if (!percentile16.values.includes("nan")) {
        scatterObjs.push({
          x: percentile16.index,
          y: percentile16.values,
          type: "scatter",
          mode: "lines",
          line: { color: "black", dash: "dash" },
          name: "16th Percentile",
          hoverinfo: "none",
          hovertemplate:
            "<b>16<sup>th</sup> percentile</b><br><br>" +
            "%{xaxis.title.text}: %{x}<br>" +
            "%{yaxis.title.text}: %{y}<extra></extra>",
        });
      }
      if (!percentile84.values.includes("nan")) {
        scatterObjs.push({
          x: percentile84.index,
          y: percentile84.values,
          type: "scatter",
          mode: "lines",
          line: { color: "black", dash: "dash" },
          name: "84th Percentile",
          hoverinfo: "none",
          hovertemplate:
            "<b>84<sup>th</sup> percentile</b><br><br>" +
            "%{xaxis.title.text}: %{x}<br>" +
            "%{yaxis.title.text}: %{y}<extra></extra>",
        });
      }
    }

    return (
      <Plot
        className={"uhs-plot"}
        data={scatterObjs}
        layout={{
          xaxis: {
            title: { text: "Period (s)" },
          },
          yaxis: {
            title: { text: "Spectral acceleration (g)" },
          },
          autosize: true,
          margin: PLOT_MARGIN,
          hovermode: "closest",
          hoverlabel: { bgcolor: "#FFF" },
          legend: {
            x: 1,
            xanchor: "right",
            y: 1,
          },
        }}
        useResizeHandler={true}
        config={{
          ...PLOT_CONFIG,
          toImageButtonOptions: {
            filename:
              extra.from === "hazard"
                ? `UHS_Plot_Lat_${String(
                    parseFloat(extra.lat).toFixed(4)
                  ).replace(".", "p")}_Lng_${String(
                    parseFloat(extra.lng).toFixed(4)
                  ).replace(".", "p")}`
                : `UHS_Plot_project_id_${extra.id}_location_${extra.location}_vs30_${extra.vs30}`,
          },
        }}
      />
    );
  }
  return <ErrorMessage />;
};

export default UHSBranchPlot;
