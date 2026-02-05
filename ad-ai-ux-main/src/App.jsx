// src/App.jsx
import React, { useState } from "react";
import { Box, Container, Paper, Typography, Divider } from "@mui/material";
import ChatLayout from "./components/ChatLayout.jsx";
import ResultViewer from "./components/ResultViewer.jsx";

function App() {
  const [lastResult, setLastResult] = useState(null);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        bgcolor: "background.default",
        py: 2,
      }}
    >
      <Container maxWidth="lg">
        <Typography variant="h4" sx={{ mb: 2, fontWeight: 600 }}>
          AI Hackathon Data Explorer
        </Typography>

        <Paper
          elevation={3}
          sx={{
            display: "flex",
            flexDirection: { xs: "column", md: "row" },
            height: { xs: "80vh", md: "70vh" },
            overflow: "hidden",
          }}
        >
          {/* Left: Chat */}
          <Box
            sx={{
              flex: 1,
              borderRight: { md: "1px solid #ddd" },
              minWidth: 0,
            }}
          >
            <ChatLayout onNewResult={setLastResult} />
          </Box>

          {/* Right: Results / charts / maps / graphs */}
          <Box
            sx={{
              flexBasis: { xs: "40%", md: "40%" },
              minWidth: { xs: "100%", md: 0 },
              display: "flex",
              flexDirection: "column",
            }}
          >
            <Typography
              variant="subtitle1"
              sx={{ px: 2, py: 1.5, fontWeight: 500 }}
            >
              Result Inspector
            </Typography>
            <Divider />
            <Box sx={{ flex: 1, p: 2, overflow: "auto" }}>
              <ResultViewer result={lastResult} />
            </Box>
          </Box>
        </Paper>
      </Container>
    </Box>
  );
}

export default App;
