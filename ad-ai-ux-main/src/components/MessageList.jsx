// src/components/MessageList.jsx
import React, { useEffect, useRef } from "react";
import {
  Box,
  List,
  ListItem,
  ListItemText,
  Chip,
  CircularProgress,
  Typography,
  Button,
  Link,
} from "@mui/material";
import BuildIcon from "@mui/icons-material/Build";

function MessageList({ messages, loading, onCreateMaintenanceNotifications }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <Box sx={{ p: 2 }}>
      <List dense sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
        {messages.map((m) => (
          <ListItem
            key={m.id}
            alignItems="flex-start"
            sx={{
              bgcolor:
                m.role === "user"
                  ? "rgba(25, 118, 210, 0.06)"
                  : "background.paper",
              borderRadius: 2,
              border: "1px solid",
              borderColor: m.isError ? "error.light" : "divider",
            }}
          >
            <ListItemText
              primary={
                <Chip
                  size="small"
                  label={m.role === "user" ? "You" : "Assistant"}
                  color={m.role === "user" ? "primary" : "default"}
                  variant="outlined"
                  sx={{ mb: 0.5 }}
                />
              }
              secondary={
                <>
                  <Typography
                    variant="body2"
                    sx={{
                      whiteSpace: "pre-wrap",
                      color: m.isError ? "error.main" : "text.primary",
                    }}
                  >
                    {m.content}
                  </Typography>
                  {m.showMaintenanceButton && (
                    <Button
                      variant="contained"
                      size="small"
                      startIcon={<BuildIcon />}
                      sx={{ mt: 1, textTransform: "none" }}
                      onClick={() => onCreateMaintenanceNotifications && onCreateMaintenanceNotifications(m)}
                    >
                      Create maintenance notification(s)
                    </Button>
                  )}
                  {m.maintenanceLinks && m.maintenanceLinks.length > 0 && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
                        Successfully created {m.maintenanceLinks.length} notification{m.maintenanceLinks.length > 1 ? 's' : ''}:
                      </Typography>
                      {m.maintenanceLinks.map((link, idx) => (
                        <Box key={idx} sx={{ ml: 2, mb: 0.5 }}>
                          <Link
                            href={link.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            sx={{ fontSize: "0.875rem" }}
                          >
                            {link.notificationNumber} - {link.vehicleRegno}
                          </Link>
                        </Box>
                      ))}
                      <Box sx={{ mt: 1 }}>
                        <Link
                          href="https://coeportal515.saphosting.de/sap/bc/ui2/flp?sap-client=600&sap-language=EN#MaintenanceNotification-display?sap-ui-tech-hint=WDA"
                          target="_blank"
                          rel="noopener noreferrer"
                          sx={{ fontSize: "0.875rem", fontWeight: 600 }}
                        >
                          🔗 Open in IE03 (Fiori Launchpad)
                        </Link>
                      </Box>
                    </Box>
                  )}
                </>
              }
            />
          </ListItem>
        ))}
        {loading && (
          <ListItem>
            <CircularProgress size={22} />
            <Typography variant="body2" sx={{ ml: 1 }}>
              Thinking…
            </Typography>
          </ListItem>
        )}
        <div ref={endRef} />
      </List>
    </Box>
  );
}

export default MessageList;
