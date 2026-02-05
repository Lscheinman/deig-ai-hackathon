// src/components/SqlTableViewer.jsx
import React from "react";
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";

/**
 * tableData = {
 *   columns: [{ id, label, property }],
 *   rows: [{ [property]: value }]
 * }
 */
function SqlTableViewer({ tableData }) {
  if (!tableData) {
    return (
      <Typography variant="body2">
        No table data available in this response.
      </Typography>
    );
  }

  const { columns, rows } = tableData;

  if (!columns.length) {
    return (
      <Typography variant="body2">
        Table structure is empty (no columns).
      </Typography>
    );
  }

  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <TableContainer
        component={Paper}
        sx={{
          flex: 1,
          maxHeight: "100%",
        }}
      >
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              {columns.map((col) => (
                <TableCell key={col.id}>{col.label}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((row, idx) => (
              <TableRow key={idx} hover>
                {columns.map((col) => (
                  <TableCell key={col.id}>
                    {String(
                      row[col.property] !== undefined
                        ? row[col.property]
                        : ""
                    )}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {!rows.length && (
        <Typography variant="body2" sx={{ mt: 1 }}>
          Table has no rows.
        </Typography>
      )}
    </Box>
  );
}

export default SqlTableViewer;
