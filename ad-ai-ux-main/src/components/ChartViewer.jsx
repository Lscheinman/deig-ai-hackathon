// src/components/ChartViewer.jsx
import React from "react";
import {
  Box,
  Paper,
  Typography,
} from "@mui/material";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

/**
 * Generic chart viewer.
 *
 * Expects `data` to be an array of objects.
 * It will:
 *  - Take the first key as X-axis (e.g. "name", "label", "category")
 *  - Take the first numeric key as Y-axis if possible,
 *    otherwise the second key.
 *
 * Example ideal shape:
 *   chartData = [
 *     { label: "Jan", value: 10 },
 *     { label: "Feb", value: 12 },
 *   ]
 */
function ChartViewer({ data }) {
  if (!Array.isArray(data) || data.length === 0) {
    return (
      <Typography variant="body2">
        No chartable data available in this response.
      </Typography>
    );
  }

  const sample = data[0];
  const keys = Object.keys(sample);

  if (keys.length < 2) {
    return (
      <Typography variant="body2">
        Not enough fields to build a chart. Expected at least 2 properties per data point.
      </Typography>
    );
  }

  // Heuristic: first key = x-axis
  const xKey = keys[0];

  // Prefer a numeric field for y-axis if possible
  let yKey = keys[1];
  for (const k of keys.slice(1)) {
    const v = sample[k];
    if (typeof v === "number") {
      yKey = k;
      break;
    }
  }

  return (
    <Paper
      sx={{
        p: 2,
        borderRadius: 2,
        border: "1px solid",
        borderColor: "divider",
        height: "100%",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <Box sx={{ mb: 1 }}>
        <Typography variant="subtitle2">
          Chart ({xKey} → {yKey})
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Using keys inferred from data. For best results, send objects like:
          {" "}
          <code>{`{ label: "Jan", value: 10 }`}</code>
        </Typography>
      </Box>

      <Box sx={{ flex: 1, minHeight: 200 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={xKey} />
            <YAxis />
            <Tooltip />
            <Line
              type="monotone"
              dataKey={yKey}
              dot={true}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  );
}

export default ChartViewer;
