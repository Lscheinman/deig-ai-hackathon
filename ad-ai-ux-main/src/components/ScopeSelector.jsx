// src/components/ScopeSelector.jsx
import React from "react";
import {
  Box,
  ToggleButtonGroup,
  ToggleButton,
  Typography,
  Tooltip,
} from "@mui/material";

function ScopeSelector({ value, onChange }) {
  const handleChange = (_event, newValue) => {
    // MUI sends null if you click the already-selected button
    if (newValue !== null) {
      onChange(newValue);
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        px: 2,
        py: 1,
        borderBottom: "1px solid",
        borderColor: "divider",
      }}
    >
      <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
        AI Scope
      </Typography>

      <Tooltip title="Choose which backend / scope to send the chat to">
        <ToggleButtonGroup
          size="small"
          exclusive
          value={value}
          onChange={handleChange}
        >
          <ToggleButton value="alpha">Alpha</ToggleButton>
          <ToggleButton value="ge">SE</ToggleButton>
          <ToggleButton value="german">DE</ToggleButton>
          <ToggleButton value="uk">UK</ToggleButton>
          <ToggleButton value="finland">FI</ToggleButton>
          <ToggleButton value="norway">NO</ToggleButton>
        </ToggleButtonGroup>
      </Tooltip>
    </Box>
  );
}

export default ScopeSelector;
