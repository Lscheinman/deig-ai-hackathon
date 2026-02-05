// src/components/ChatLayout.jsx
import React, { useState } from "react";
import { Box, Divider, Button, Chip, Switch, FormControlLabel } from "@mui/material";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import AnalyticsIcon from "@mui/icons-material/Analytics";
import { sendChatMessage, getSessionId, clearSession, triggerPalSummary, createMaintenanceNotification } from "../api.js";
import MessageList from "./MessageList.jsx";
import ChatInput from "./ChatInput.jsx";
import ScopeSelector from "./ScopeSelector.jsx";

function ChatLayout({ onNewResult }) {
  const [messages, setMessages] = useState([
    {
      id: "sys-1",
      role: "assistant",
      content:
        "Ask me something, then change the AI scope above to route to Alpha, SE (Swedish), DE (German), UK, FI (Finland), or NO (Norway).",
    },
  ]);

  const [loading, setLoading] = useState(false);
  const [scope, setScope] = useState("alpha"); // "alpha" | "ge" (swedish) | "german" | "uk" | "finland"
  const [hasSession, setHasSession] = useState(false);
  const [ukSwitchEnabled, setUkSwitchEnabled] = useState(true);

  const handleScopeChange = (newScope) => {
    setScope(newScope);
    // Check if new scope has an active session
    setHasSession(!!getSessionId(newScope));
  };

  const handleClearSession = () => {
    clearSession(scope);
    setHasSession(false);
    setMessages([
      {
        id: "sys-cleared",
        role: "assistant",
        content: "Conversation cleared. I won't remember our previous exchanges.",
      },
    ]);
  };

  const handlePalAnalysis = async () => {
    setLoading(true);

    const assistantId = `a-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      {
        id: assistantId,
        role: "assistant",
        content: "Starting PAL clustering analysis...",
        isStreaming: true,
        isStatus: true,
      },
    ]);

    let streamedContent = "";

    try {
      const onStream = (type, content) => {
        if (type === "status") {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content, isStatus: true }
                : msg
            )
          );
        } else if (type === "text") {
          streamedContent += content;
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content: streamedContent, isStatus: false }
                : msg
            )
          );
        }
      };

      const response = await triggerPalSummary(onStream);

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                content: response.reply || "(no reply text from PAL analysis)",
                isStreaming: false,
                isStatus: false,
                raw: response,
              }
            : msg
        )
      );

      if (onNewResult) {
        onNewResult(response);
      }
    } catch (err) {
      setMessages((prev) => {
        const withoutPlaceholder = prev.filter((msg) => msg.id !== assistantId);
        return [
          ...withoutPlaceholder,
          {
            id: `e-${Date.now()}`,
            role: "assistant",
            content: `PAL Analysis Error: ${err.message}`,
            isError: true,
          },
        ];
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSend = async (text) => {
    if (!text.trim()) return;

    const userMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: text,
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    // Create a placeholder for the assistant response
    const assistantId = `a-${Date.now()}`;
    let streamedContent = "";
    let lastStatus = "";

    // Add initial placeholder message
    setMessages((prev) => [
      ...prev,
      {
        id: assistantId,
        role: "assistant",
        content: "",
        isStreaming: scope === "german" || scope === "ge" || scope === "uk" || scope === "finland",
      },
    ]);

    try {
      // Streaming callback for German API
      const onStream = (type, content) => {
        if (type === "status") {
          // Temporary status message
          lastStatus = content;
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content: `_${content}_`, isStatus: true }
                : msg
            )
          );
        } else if (type === "text") {
          // Accumulate streaming text
          streamedContent += content;
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content: streamedContent, isStatus: false }
                : msg
            )
          );
        }
      };

      // Pass current scope and streaming callback into the API call
      const response = await sendChatMessage(text, scope, onStream, {
        chatty: scope === "uk" ? ukSwitchEnabled : true,
      });

      // Update final message with complete response
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                content: response.reply || "(no reply text from API)",
                isStreaming: false,
                isStatus: false,
                raw: response,
                showMaintenanceButton: response._hasMaintenanceData || false,
              }
            : msg
        )
      );

      // Update session state after receiving response
      setHasSession(!!getSessionId(scope));

      if (onNewResult) {
        // include scope in the result passed to the right-hand panel
        onNewResult(response);
      }
    } catch (err) {
      // Update the message with error or create new error message
      setMessages((prev) => {
        const withoutPlaceholder = prev.filter((msg) => msg.id !== assistantId);
        return [
          ...withoutPlaceholder,
          {
            id: `e-${Date.now()}`,
            role: "assistant",
            content: `Error (${scope}): ${err.message}`,
            isError: true,
          },
        ];
      });
    } finally {
      setLoading(false);
    }
  };

  const handleCreateMaintenanceNotifications = async (message) => {
    if (!message.raw || !message.raw._raw || !message.raw._raw.results) {
      console.error("No vehicle data found in message");
      return;
    }

    const vehicleData = message.raw._raw.results;
    setLoading(true);

    try {
      const results = [];
      
      // Create maintenance notification for each vehicle
      for (const vehicle of vehicleData) {
        const result = await createMaintenanceNotification(vehicle);
        results.push({
          notificationNumber: result.notification_number,
          vehicleRegno: result.vehicle_regno,
          url: `https://coeportal515.saphosting.de/sap/opu/odata/sap/API_MAINTNOTIFICATION/MaintenanceNotification('${result.notification_number}')?$format=json`,
        });
      }

      // Add success message with links
      setMessages((prev) => [
        ...prev,
        {
          id: `success-${Date.now()}`,
          role: "assistant",
          content: `Successfully created maintenance notifications for ${results.length} vehicle${results.length > 1 ? 's' : ''}.`,
          maintenanceLinks: results,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: "assistant",
          content: `Failed to create maintenance notifications: ${error.message}`,
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Scope selector at the top */}
      <ScopeSelector value={scope} onChange={handleScopeChange} />

      {/* Session status and controls */}
      <Box
        sx={{
          px: 2,
          py: 1,
          display: "flex",
          alignItems: "center",
          gap: 1,
          borderBottom: "1px solid rgba(0, 0, 0, 0.12)",
        }}
      >
        {hasSession ? (
          <>
            <Chip
              label="Active Conversation"
              color="primary"
              size="small"
              variant="outlined"
            />
            <Chip
              label="Remembers last 5 exchanges"
              size="small"
              sx={{ fontSize: "0.7rem" }}
            />
            <Box sx={{ flex: 1 }} />
            {scope === "ge" && (
              <Button
                size="small"
                startIcon={<AnalyticsIcon />}
                onClick={handlePalAnalysis}
                variant="contained"
                disabled={loading}
                sx={{ textTransform: "none", mr: 1 }}
              >
                PAL
              </Button>
            )}
            {scope === "uk" && (
              <FormControlLabel
                control={
                  <Switch
                    checked={ukSwitchEnabled}
                    onChange={(e) => setUkSwitchEnabled(e.target.checked)}
                    size="small"
                  />
                }
                label={ukSwitchEnabled ? "Chatty" : "Less chatty"}
                sx={{ mr: 1 }}
              />
            )}
            <Button
              size="small"
              startIcon={<RestartAltIcon />}
              onClick={handleClearSession}
              variant="outlined"
              sx={{ textTransform: "none" }}
            >
              New Conversation
            </Button>
          </>
        ) : (
          <>
            <Chip
              label="Fresh conversation - no previous context"
              size="small"
              variant="outlined"
            />
            <Box sx={{ flex: 1 }} />
            {scope === "ge" && (
              <Button
                size="small"
                startIcon={<AnalyticsIcon />}
                onClick={handlePalAnalysis}
                variant="contained"
                disabled={loading}
                sx={{ textTransform: "none" }}
              >
                PAL
              </Button>
            )}
            {scope === "uk" && (
              <FormControlLabel
                control={
                  <Switch
                    checked={ukSwitchEnabled}
                    onChange={(e) => setUkSwitchEnabled(e.target.checked)}
                    size="small"
                  />
                }
                label={ukSwitchEnabled ? "Chatty" : "Less chatty"}
              />
            )}
          </>
        )}
      </Box>

      {/* Message list */}
      <Box sx={{ flex: 1, overflow: "auto" }}>
        <MessageList 
          messages={messages} 
          loading={loading} 
          onCreateMaintenanceNotifications={handleCreateMaintenanceNotifications}
        />
      </Box>

      <Divider />

      {/* Input area */}
      <Box sx={{ p: 1.5 }}>
        <ChatInput onSend={handleSend} disabled={loading} scope={scope} />
      </Box>
    </Box>
  );
}

export default ChatLayout;
