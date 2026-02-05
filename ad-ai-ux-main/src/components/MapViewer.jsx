// src/components/MapViewer.jsx
import React, { useEffect } from "react";
import { Box, Paper, Typography } from "@mui/material";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Fix for default marker icons in production builds
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
});

// Component to fit bounds when points change
function FitBounds({ points }) {
  const map = useMap();
  
  useEffect(() => {
    if (points && points.length > 0) {
      const bounds = points.map(p => [p.lat, p.lon]);
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [points, map]);
  
  return null;
}

function MapViewer({ data }) {
  if (!data || !Array.isArray(data.points) || data.points.length === 0) {
    return (
      <Typography variant="body2">
        No mappable locations found in this response.
      </Typography>
    );
  }

  const points = data.points;

  // Basic center: average of all points, fallback to first point
  let centerLat = points[0].lat;
  let centerLon = points[0].lon;
  if (points.length > 1) {
    centerLat =
      points.reduce((sum, p) => sum + p.lat, 0) / points.length;
    centerLon =
      points.reduce((sum, p) => sum + p.lon, 0) / points.length;
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
        <Typography variant="subtitle2">Geographic distribution</Typography>
        <Typography variant="caption" color="text.secondary">
          Showing {points.length} location
          {points.length !== 1 ? "s" : ""} inferred from the result data.
        </Typography>
      </Box>

      <Box sx={{ flex: 1, minHeight: 250 }}>
        <MapContainer
          center={[centerLat, centerLon]}
          zoom={2}
          style={{ height: "100%", width: "100%" }}
          scrollWheelZoom={false}
        >
          <TileLayer
            attribution='&copy; OpenStreetMap contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FitBounds points={points} />
          {points.map((p, idx) => (
            <Marker key={idx} position={[p.lat, p.lon]}>
              {p.label && <Popup>{p.label}</Popup>}
            </Marker>
          ))}
        </MapContainer>
      </Box>
    </Paper>
  );
}

export default MapViewer;
