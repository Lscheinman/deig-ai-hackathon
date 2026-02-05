// src/components/ResultViewer.jsx
import React from "react";
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Paper,
  Alert,
  Stack,
  Chip,
} from "@mui/material";
import SqlTableViewer from "./SqlTableViewer.jsx";
import ChartViewer from "./ChartViewer.jsx";
import MapViewer from "./MapViewer.jsx";

function a11yProps(index) {
  return {
    id: `result-tab-${index}`,
    "aria-controls": `result-tabpanel-${index}`,
  };
}

function TabPanel({ children, value, index }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`result-tabpanel-${index}`}
      aria-labelledby={`result-tab-${index}`}
      style={{ height: "100%" }}
    >
      {value === index && <Box sx={{ pt: 1, height: "100%" }}>{children}</Box>}
    </div>
  );
}

function ResultViewer({ result }) {
  const [tab, setTab] = React.useState(0);

  if (!result) {
    return (
      <Alert severity="info">
        Send a prompt that returns data, and you’ll see structured results
        here — charts, maps, tables, or graphs.
      </Alert>
    );
  }

  const hasChart = !!result.chartData;
  const hasMap = !!result.mapData;
  const hasGraph = !!result.graphData;
  const hasTable = !!result.tableData;

  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Tabs
        value={tab}
        onChange={(_, v) => setTab(v)}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ minHeight: 40 }}
      >
        <Tab label="Summary" {...a11yProps(0)} />
        <Tab label="Table" {...a11yProps(1)} disabled={!hasTable} />
        <Tab label="Chart" {...a11yProps(2)} disabled={!hasChart} />
        <Tab label="Map" {...a11yProps(3)} disabled={!hasMap} />
        <Tab label="Graph" {...a11yProps(4)} disabled={!hasGraph} />
        <Tab label="Raw JSON" {...a11yProps(5)} />
      </Tabs>

      <Box sx={{ flex: 1, mt: 1, overflow: "auto" }}>
        {/* Summary */}
        <TabPanel value={tab} index={0}>
          <Stack spacing={1}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              {result._scope && (
                <Chip
                  size="small"
                  label={`Scope: ${result._scope}`}
                  variant="outlined"
                />
              )}
              {result.sqlQuery && (
                <Chip
                  size="small"
                  label="SQL mode"
                  color="secondary"
                  variant="outlined"
                />
              )}
            </Box>

            <Typography variant="body2">
              {result.reply || "No summary text returned by API."}
            </Typography>

            {result.sqlQuery && (
              <Paper
                sx={{
                  mt: 1,
                  p: 1.5,
                  bgcolor: "#111",
                  color: "#eee",
                  fontFamily: "monospace",
                  fontSize: 12,
                }}
              >
                <Typography
                  variant="caption"
                  sx={{ display: "block", mb: 0.5, color: "#bbb" }}
                >
                  SQL Query
                </Typography>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                  {result.sqlQuery}
                </pre>
              </Paper>
            )}

            <Stack spacing={1} sx={{ mt: 1 }}>
              {hasTable && (
                <Alert severity="success">
                  Tabular result available. See the <b>Table</b> tab.
                </Alert>
              )}
              {hasChart && (
                <Alert severity="info">
                  Chart data available ({Array.isArray(result.chartData)
                    ? result.chartData.length
                    : "object"}
                  ). Hook Recharts here.
                </Alert>
              )}
              {hasMap && (
                <Alert severity="info">
                  Map data available. Hook React Leaflet / Mapbox in here.
                </Alert>
              )}
              {hasGraph && (
                <Alert severity="info">
                  Graph data available. Hook React Flow / Cytoscape here.
                </Alert>
              )}
            </Stack>
          </Stack>
        </TabPanel>

        {/* Table */}
        <TabPanel value={tab} index={1}>
          <SqlTableViewer tableData={result.tableData} />
        </TabPanel>

        {/* Chart placeholder */}
        <TabPanel value={tab} index={2}>
        {/* Chart – now real */}
          <ChartViewer data={result.chartData} />
        </TabPanel>

        {/* Map placeholder */}
        <TabPanel value={tab} index={3}>
          <MapViewer data={result.mapData} />
        </TabPanel>

        {/* Graph placeholder */}
        <TabPanel value={tab} index={4}>
          <Paper
            sx={{
              p: 2,
              borderRadius: 2,
              border: "1px dashed",
              borderColor: "divider",
            }}
          >
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Network graph placeholder
            </Typography>
            <Typography variant="body2">
              Mount React Flow / Cytoscape here using{" "}
              <code>result.graphData</code>.
            </Typography>
          </Paper>
        </TabPanel>

        {/* Raw JSON */}
        <TabPanel value={tab} index={5}>
          <Paper
            sx={{
              p: 2,
              borderRadius: 2,
              bgcolor: "#111",
              color: "#eee",
              fontFamily: "monospace",
              fontSize: 12,
            }}
          >
            <pre style={{ margin: 0 }}>
              {JSON.stringify(result._raw ?? result, null, 2)}
            </pre>
          </Paper>
        </TabPanel>
      </Box>
    </Box>
  );
}

export default ResultViewer;
