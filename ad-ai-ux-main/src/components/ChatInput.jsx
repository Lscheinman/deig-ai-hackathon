// src/components/ChatInput.jsx
import React, { useState, useRef } from "react";
import { Box, TextField, IconButton, Paper, CircularProgress } from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import AttachFileIcon from "@mui/icons-material/AttachFile";
import { uploadPdf } from "../api.js";

function ChatInput({ onSend, disabled, scope }) {
  const [value, setValue] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setValue("");
  };

  const handleFileClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.type !== "application/pdf") {
      console.warn("Only PDF files are supported");
      return;
    }

    setUploading(true);
    try {
      console.log("Uploading PDF:", file.name);
      const result = await uploadPdf(file);
      console.log("PDF upload result:", result);
      
      // Extract text content from result and send to UK agent
      const extractedText = result.text || result.content || result.extracted_text || "";
      if (extractedText) {
        const prompt = `Analyze which materials are needed for this operational order and which predicted demands are likely to be affected by it in turn:\n\n${extractedText}`;
        onSend(prompt);
      } else {
        console.warn("No text extracted from PDF");
      }
    } catch (error) {
      console.error("PDF upload failed:", error);
    } finally {
      setUploading(false);
    }

    // Reset file input
    e.target.value = "";
  };

  return (
    <Paper
      component="form"
      onSubmit={handleSubmit}
      sx={{
        display: "flex",
        alignItems: "center",
        p: 1,
        borderRadius: 3,
      }}
      elevation={2}
    >
      <TextField
        fullWidth
        variant="outlined"
        size="small"
        placeholder="Ask a question or describe the data you want…"
        value={value}
        disabled={disabled}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            handleSubmit(e);
          }
        }}
      />
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />
      <Box sx={{ ml: 1, display: "flex", gap: 0.5 }}>
        {scope === "uk" && (
          <IconButton
            color="primary"
            disabled={disabled || uploading}
            onClick={handleFileClick}
            title="Upload PDF"
          >
            {uploading ? (
              <CircularProgress size={24} />
            ) : (
              <AttachFileIcon />
            )}
          </IconButton>
        )}
        <IconButton
          type="submit"
          color="primary"
          disabled={disabled || !value.trim() || uploading}
        >
          <SendIcon />
        </IconButton>
      </Box>
    </Paper>
  );
}

export default ChatInput;
