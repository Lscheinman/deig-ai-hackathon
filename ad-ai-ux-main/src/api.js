// src/api.js

const API_URL = "https://ad-hackathon-api.cfapps.ap10.hana.ondemand.com/";
const GERMAN_API_URL = "https://oslo-api.cfapps.ap10.hana.ondemand.com/api/german/warehouse/stream";
const SWEDISH_API_URL = "https://oslo-api.cfapps.ap10.hana.ondemand.com/api/sweden/suppliers/stream";
const UK_API_URL = "https://oslo-api.cfapps.ap10.hana.ondemand.com/api/uk/demand/stream";
const FINLAND_API_URL = "https://oslo-api.cfapps.ap10.hana.ondemand.com/api/finland/health/stream";
const NORWAY_API_URL = "https://oslo-api.cfapps.ap10.hana.ondemand.com/api/norway/vehicle/engine-oil-temp";

// Map scopes to agent types for session management
const SCOPE_TO_AGENT = {
  ge: 'sweden_suppliers',
  german: 'german_warehouse',
  uk: 'uk_demand',
  finland: 'finland_health',
};

// Session ID storage per agent
const sessionIds = {
  german_warehouse: null,
  sweden_suppliers: null,
  uk_demand: null,
  finland_health: null,
};

// Load session IDs from localStorage on init
Object.keys(sessionIds).forEach(agent => {
  const stored = localStorage.getItem(`session-${agent}`);
  if (stored) sessionIds[agent] = stored;
});

// Map logical scopes to endpoint paths
const API_CONFIG = {
  alpha: {
    endpoint: "alpha",
    messageKey: "message",
    method: "POST",
  },
  ge: {
    endpoint: SWEDISH_API_URL,
    messageKey: "query",
    method: "STREAM",
  },
  german: {
    endpoint: GERMAN_API_URL,
    messageKey: "question",
    method: "STREAM",
  },
  uk: {
    endpoint: UK_API_URL,
    messageKey: "question",
    method: "STREAM",
  },
  finland: {
    endpoint: FINLAND_API_URL,
    messageKey: "question",
    method: "STREAM",
  },
  norway: {
    endpoint: NORWAY_API_URL,
    messageKey: "query",
    method: "GET",
  },
};

const US_STATE_CENTROIDS = [
  { state: "California", lat: 36.7783, lon: -119.4179 },
  { state: "Texas", lat: 31.9686, lon: -99.9018 },
  { state: "Florida", lat: 27.6648, lon: -81.5158 },
  { state: "New York", lat: 43.0000, lon: -75.0000 },
  { state: "Illinois", lat: 40.0000, lon: -89.0000 },
  { state: "Georgia", lat: 32.1656, lon: -82.9001 },
  { state: "North Carolina", lat: 35.7596, lon: -79.0193 },
  { state: "Washington", lat: 47.7511, lon: -120.7401 },
  { state: "Colorado", lat: 39.1130, lon: -105.3589 },
  { state: "Arizona", lat: 34.0489, lon: -111.0937 }
];

/**
 * Handle streaming Server-Sent Events (SSE) response from streaming APIs
 */
async function handleStreamResponse(response, onStream, scope) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completeData = null;
  let accumulatedContent = ""; // Track accumulated streaming content

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep incomplete line in buffer

      for (const line of lines) {
        if (!line.trim() || !line.startsWith("data: ")) continue;

        const jsonStr = line.substring(6); // Remove "data: " prefix
        
        // Handle [DONE] marker from Swedish API
        if (jsonStr === "[DONE]") continue;
        
        try {
          const event = JSON.parse(jsonStr);

          if (event.type === "status" && onStream) {
            // Temporary status message (German API)
            onStream("status", event.message);
          } else if (event.type === "text" && onStream) {
            // Stream text chunk (German API)
            onStream("text", event.content);
            accumulatedContent += event.content;
          } else if (event.type === "content" && onStream) {
            // Stream content chunk (Swedish API / PAL)
            onStream("text", event.content);
            accumulatedContent += event.content;
          } else if (event.type === "complete") {
            // Final complete message with all data (German API)
            completeData = event.data;
          } else if (event.type === "metadata") {
            // Final metadata with results (Swedish API / PAL)
            completeData = event;
          }
        } catch (e) {
          console.warn("Failed to parse SSE event:", line, e);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  if (!completeData) {
    throw new Error("Stream ended without complete data");
  }

  // For PAL analysis, include accumulated content
  if (scope === "pal" && accumulatedContent) {
    completeData._accumulatedContent = accumulatedContent;
  }

  // Normalize the complete data
  return normalizeResponse(completeData, scope);
}

/**
 * Normalize API response data into a consistent format
 */
function normalizeResponse(data, scope) {
  // Special handling for Norway vehicle endpoint
  if (scope === "norway") {
    let tableData = null;
    if (data.results && Array.isArray(data.results) && data.results.length > 0) {
      const firstRow = data.results[0];
      const columns = Object.keys(firstRow).map((key) => ({
        id: key,
        label: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        property: key,
      }));
      tableData = { columns, rows: data.results };
    }

    return {
      reply: data.note || "(no note from vehicle analysis)",
      chartData: null,
      mapData: null,
      graphData: null,
      tableData,
      sqlQuery: null,
      _scope: scope,
      _raw: data,
      _rowCount: data.row_count,
      _timestamp: data.timestamp,
      _hasMaintenanceData: data.results && data.results.length > 0,
    };
  }

  // Special handling for PAL analysis
  if (scope === "pal") {
    // Build table from cluster_profiles
    let tableData = null;
    if (data.cluster_profiles && Array.isArray(data.cluster_profiles)) {
      const columns = [
        { id: "CLUSTER_ID", label: "Cluster", property: "CLUSTER_ID" },
        { id: "COUNT", label: "Suppliers", property: "COUNT" },
        { id: "AVG_SCORE", label: "Avg Score", property: "AVG_SCORE" },
        { id: "AVG_VOLUME", label: "Avg Volume", property: "AVG_VOLUME" },
        { id: "AVG_LATITUDE", label: "Latitude", property: "AVG_LATITUDE" },
        { id: "AVG_LONGITUDE", label: "Longitude", property: "AVG_LONGITUDE" },
      ];
      tableData = { columns, rows: data.cluster_profiles };
    }

    // Build map from supplier_assignments
    let mapData = null;
    if (data.supplier_assignments && Array.isArray(data.supplier_assignments)) {
      const points = data.supplier_assignments
        .filter(s => s.LATITUDE && s.LONGITUDE)
        .map(s => ({
          lat: typeof s.LATITUDE === 'number' ? s.LATITUDE : parseFloat(s.LATITUDE),
          lon: typeof s.LONGITUDE === 'number' ? s.LONGITUDE : parseFloat(s.LONGITUDE),
          label: `${s.SUPPLIER_NAME} (Cluster ${s.CLUSTER_ID}, Score: ${s.COMBINED_SCORE})`,
        }));
      if (points.length > 0) {
        mapData = { points };
      }
    }

    // Build chart from cluster_profiles
    let chartData = null;
    if (data.cluster_profiles && Array.isArray(data.cluster_profiles)) {
      chartData = data.cluster_profiles.map(c => ({
        Cluster: `Cluster ${c.CLUSTER_ID}`,
        Score: c.AVG_SCORE,
        Volume: c.AVG_VOLUME,
        Suppliers: c.COUNT,
      }));
    }

    return {
      reply: data.summary || data._accumulatedContent || "(no PAL analysis content)",
      chartData,
      mapData,
      graphData: null,
      tableData,
      sqlQuery: null,
      _scope: scope,
      _raw: data,
      _metadata: data.clustering_metadata || null,
      _clusterProfiles: data.cluster_profiles || null,
      _supplierAssignments: data.supplier_assignments || null,
      _model: data.model_used || null,
    };
  }

  // --- Normalization logic for other endpoints ---
  // 1. reply: prefer description.reply, then reply, then response (German API), then fallback
  const replyText =
    data.description?.reply ??
    data.reply ??
    data.response ??
    "(no reply text in response)";

  // 2. tableData: normalized view of sql_result.table if present, or German API results
  let tableData = null;
  if (data.sql_result?.table) {
    const table = data.sql_result.table;
    const columns =
      table.columns?.map((col, idx) => ({
        id: col.property || col.label || `col_${idx}`,
        label: col.label || col.property || `Column ${idx + 1}`,
        property: col.property || col.label || `col_${idx}`,
      })) || [];
    const rows = Array.isArray(table.rows) ? table.rows : [];
    tableData = { columns, rows };
  } else if (data.results && Array.isArray(data.results) && data.results.length > 0) {
    // German API format: array of result objects
    const firstRow = data.results[0];
    const columns = Object.keys(firstRow).map((key) => ({
      id: key,
      label: key,
      property: key,
    }));
    tableData = { columns, rows: data.results };
  }

  // 3. chartData
  // Prefer explicit chartData from backend, else derive from tableData
  let chartData = data.chartData ?? null;

  if (!chartData && tableData && tableData.rows.length > 0) {
    const { columns, rows } = tableData;
    const firstRow = rows[0];

    // Separate numeric vs string columns based on first row
    const numericCols = columns.filter((c) =>
      typeof firstRow[c.property] === "number"
    );
    const stringCols = columns.filter((c) =>
      typeof firstRow[c.property] === "string"
    );

    if (numericCols.length > 0 && stringCols.length > 0) {
      // Pick label column: prefer *DESCRIPTION*, else first string column
      let labelCol = stringCols[0];
      const descCandidate = stringCols.find((c) =>
        (c.label || c.property || "").toUpperCase().includes("DESCRIPTION")
      );
      if (descCandidate) {
        labelCol = descCandidate;
      }

      const valueCol = numericCols[0];

      chartData = rows.map((row) => ({
        label: row[labelCol.property],
        value: row[valueCol.property],
      }));
    }
  }

  // 4. mapData – backend may send it, otherwise infer from tableData
  let mapData = data.mapData ?? null;

  if (!mapData && tableData && tableData.rows.length > 0) {
    const { columns, rows } = tableData;
    const firstRow = rows[0];

    const normalizeName = (c) =>
      (c.label || c.property || "").toUpperCase();

    // Helper to check if a value can be used as a coordinate (number or parseable string)
    const isCoordinate = (v) => {
      if (typeof v === "number") return !Number.isNaN(v);
      if (typeof v === "string") {
        const num = parseFloat(v);
        return !Number.isNaN(num);
      }
      return false;
    };

    const latCol = columns.find((c) => {
      const name = normalizeName(c);
      const v = firstRow[c.property];
      return (
        isCoordinate(v) &&
        (name === "LAT" || 
         name === "LATITUDE" || 
         name.includes("LAT") || 
         name.includes("LATITUDE"))
      );
    });

    const lonCol = columns.find((c) => {
      const name = normalizeName(c);
      const v = firstRow[c.property];
      return (
        isCoordinate(v) &&
        (name === "LON" ||
          name === "LONG" ||
          name === "LNG" ||
          name === "LONGITUDE" ||
          name.includes("LON") ||
          name.includes("LNG") ||
          name.includes("LONG") ||
          name.includes("LONGITUDE"))
      );
    });

    const stringCols = columns.filter(
      (c) => typeof firstRow[c.property] === "string"
    );

    // Case 1: explicit lat/lon columns
    if (latCol && lonCol) {
      const labelCol =
        stringCols.find((c) =>
          normalizeName(c).includes("DESCRIPTION")
        ) || stringCols[0] || null;

      const points = rows
        .map((row) => {
          const latVal = row[latCol.property];
          const lonVal = row[lonCol.property];
          // Convert to numbers if they're strings
          const lat = typeof latVal === "string" ? parseFloat(latVal) : latVal;
          const lon = typeof lonVal === "string" ? parseFloat(lonVal) : lonVal;
          
          return {
            lat,
            lon,
            label: labelCol ? row[labelCol.property] : undefined,
          };
        })
        // filter out any junk
        .filter(
          (p) =>
            typeof p.lat === "number" &&
            typeof p.lon === "number" &&
            !Number.isNaN(p.lat) &&
            !Number.isNaN(p.lon)
        );

      if (points.length) {
        mapData = { points };
      }
    } else {
      // Case 2: country-based mapping (very rough)
      const countryCol = columns.find((c) =>
        /(COUNTRY|NATION)/i.test(c.label || c.property || "")
      );

      if (countryCol) {
        const COUNTRY_COORDS = {
          USA: { lat: 37.1, lon: -95.7 },
          "UNITED STATES": { lat: 37.1, lon: -95.7 },
          GERMANY: { lat: 51.0, lon: 10.0 },
          FRANCE: { lat: 46.2, lon: 2.2 },
          "UNITED KINGDOM": { lat: 55.0, lon: -3.0 },
          UK: { lat: 55.0, lon: -3.0 },
          CANADA: { lat: 56.1, lon: -106.3 },
          AUSTRALIA: { lat: -25.0, lon: 133.0 },
          JAPAN: { lat: 36.2, lon: 138.3 },
        };

        const points = [];

        for (const row of rows) {
          const raw = row[countryCol.property];
          if (!raw) continue;
          const key = String(raw).trim().toUpperCase();
          const coord = COUNTRY_COORDS[key];
          if (coord) {
            points.push({
              lat: coord.lat,
              lon: coord.lon,
              label: raw,
            });
          }
        }

        if (points.length) {
          mapData = { points };
        }
      }
    }
  }
  // Don't show default US states if no map data found
  // mapData remains null if no coordinates were found

  return {
    reply: replyText,
    chartData,
    mapData: mapData,
    graphData: data.graphData ?? null,
    tableData,
    sqlQuery: data.sql_query || null,
    _scope: scope,
    _raw: data,
  };
}

/**
 * Get session ID for a scope
 */
export function getSessionId(scope) {
  const agentType = SCOPE_TO_AGENT[scope];
  return agentType ? sessionIds[agentType] : null;
}

/**
 * Clear session for a scope
 */
export function clearSession(scope) {
  const agentType = SCOPE_TO_AGENT[scope];
  if (agentType) {
    sessionIds[agentType] = null;
    localStorage.removeItem(`session-${agentType}`);
  }
}

/**
 * Send a chat message to the backend.
 *
 * @param {string} message
 * @param {"alpha"|"ge"|"german"|"uk"|"finland"} scope
 * @param {Function} onStream - Optional callback for streaming updates: (type, content) => void
 * @param {Object} options - Optional configuration { chatty: boolean }
 */
export async function sendChatMessage(message, scope = "alpha", onStream = null, options = {}) {
  const cfg = API_CONFIG[scope];
  if (!cfg) throw new Error(`Unknown scope: ${scope}`);

  const agentType = SCOPE_TO_AGENT[scope];
  const { chatty = true } = options;
  let url, fetchOptions;

  if (cfg.method === "STREAM") {
    // Streaming APIs use GET with query parameters
    // All streaming endpoints use gpt-5
    const model = "gpt-5";
    
    // For UK in less chatty mode, append brief instruction
    let actualMessage = message;
    if (scope === "uk" && !chatty) {
      actualMessage = `${message}\n\nBe brief in your response, max 250 characters`;
    }
    
    const params = new URLSearchParams({
      [cfg.messageKey]: actualMessage,
      model: model
    });
    url = `${cfg.endpoint}?${params.toString()}`;
    
    const headers = {
      accept: "application/json",
    };
    
    // Add session ID if we have one for this agent
    if (agentType && sessionIds[agentType]) {
      headers['X-Session-ID'] = sessionIds[agentType];
    }
    
    fetchOptions = {
      method: "GET",
      headers,
    };
  } else if (cfg.method === "GET") {
    // Regular GET with query parameters
    const params = new URLSearchParams({
      [cfg.messageKey]: message,
      model: "gpt-4o"
    });
    url = `${cfg.endpoint}?${params.toString()}`;
    fetchOptions = {
      method: "GET",
      headers: {
        accept: "application/json",
      },
    };
  } else {
    // Original POST method for alpha and ge
    const body = new URLSearchParams({
      [cfg.messageKey]: message,
    }).toString();
    url = API_URL + cfg.endpoint;
    fetchOptions = {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        accept: "application/json",
      },
      body,
    };
  }

  const res = await fetch(url, fetchOptions);

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Chat API error: ${res.status} ${text}`);
  }

  // Capture session ID from response headers
  if (agentType) {
    const newSessionId = res.headers.get('X-Session-ID');
    if (newSessionId) {
      sessionIds[agentType] = newSessionId;
      localStorage.setItem(`session-${agentType}`, newSessionId);
    }
  }

  // Handle streaming response for streaming APIs
  if (cfg.method === "STREAM") {
    return await handleStreamResponse(res, onStream, scope);
  }

  const data = await res.json();
  return normalizeResponse(data, scope);
}

// Example stub for future: get graph data directly
export async function fetchGraphById(graphId) {
  throw new Error("fetchGraphById is not implemented yet.");
}

/**
 * Upload PDF file to pdf_reader endpoint
 * @param {File} file - PDF file to upload
 */
export async function uploadPdf(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("https://oslo-api.cfapps.ap10.hana.ondemand.com/pdf_reader", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`PDF upload error: ${res.status} ${text}`);
  }

  return await res.json();
}

/**
 * Trigger PAL clustering analysis for Swedish suppliers
 * @param {Function} onStream - Optional callback for streaming updates
 */
export async function triggerPalSummary(onStream = null) {
  const url = "https://oslo-api.cfapps.ap10.hana.ondemand.com/api/sweden/pal/summary";
  
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "accept": "application/json",
    },
    body: JSON.stringify({ model: "gpt-5" }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`PAL analysis error: ${res.status} ${text}`);
  }

  // Handle streaming response (same format as Swedish API)
  return await handleStreamResponse(res, onStream, "pal");
}

/**
 * Create maintenance notification for a vehicle
 * @param {Object} vehicleData - Vehicle data from Norway endpoint
 */
export async function createMaintenanceNotification(vehicleData) {
  const url = "https://oslo-api.cfapps.ap10.hana.ondemand.com/api/norway/maintenance-notification";
  
  const payload = {
    vehicle_regno: vehicleData.vehicle_regno,
    tripname: vehicleData.tripname,
    category: vehicleData.category,
    duration_minutes: parseFloat(vehicleData.agg_duration_dec_min),
    category_factor_product: parseFloat(vehicleData.category_factor_product),
    notification_text: `High engine oil temp - Vehicle ${vehicleData.vehicle_regno}`,
    priority: "3",
  };

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "accept": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Maintenance notification error: ${res.status} ${text}`);
  }

  return await res.json();
}
