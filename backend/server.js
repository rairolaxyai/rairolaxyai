
"use strict";

require("dotenv").config();

const express = require("express");
const cors = require("cors");
const crypto = require("crypto");

const app = express();

const PORT = Number(process.env.PORT) || 10000;
const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;
const NVIDIA_MODEL =
  process.env.NVIDIA_MODEL || "nvidia/nemotron-3-super-120b-a12b";
const NVIDIA_API_URL =
  "https://integrate.api.nvidia.com/v1/chat/completions";

app.disable("x-powered-by");
app.use(cors());
app.use(express.json({ limit: "2mb" }));

// PostgreSQL temporarily disabled.
// It can be added again later without changing the NVIDIA chat API.

function makeId() {
  return crypto.randomUUID();
}

function sendError(res, status, message) {
  return res.status(status).json({
    success: false,
    error: message
  });
}

function databaseUnavailable(res) {
  return sendError(
    res,
    503,
    "Conversation storage is temporarily disabled. PostgreSQL will be added later."
  );
}

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) return [];

  return messages
    .filter(
      (message) =>
        message &&
        ["system", "user", "assistant"].includes(message.role) &&
        typeof message.content === "string"
    )
    .slice(-40)
    .map((message) => ({
      role: message.role,
      content: message.content.slice(0, 20000)
    }));
}

// Basic route
app.get("/", (req, res) => {
  res.json({
    success: true,
    name: "Rairolaxy AI Backend",
    status: "running",
    database: "temporarily_disabled",
    endpoints: ["/health", "/api/status", "/api/chat"]
  });
});

// Health check does not depend on PostgreSQL.
app.get("/health", (req, res) => {
  res.status(200).json({
    success: true,
    status: "healthy",
    database: "disabled_temporarily",
    aiConfigured: Boolean(NVIDIA_API_KEY),
    provider: "NVIDIA",
    model: NVIDIA_MODEL
  });
});

app.get("/api/status", (req, res) => {
  res.json({
    success: true,
    backend: "online",
    databaseConfigured: false,
    databaseReady: false,
    aiConfigured: Boolean(NVIDIA_API_KEY),
    provider: "NVIDIA",
    model: NVIDIA_MODEL
  });
});

// NVIDIA AI request
async function getNvidiaReply(messages) {
  if (!NVIDIA_API_KEY) {
    const error = new Error(
      "NVIDIA_API_KEY is not configured on the server."
    );
    error.status = 503;
    throw error;
  }

  const response = await fetch(NVIDIA_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${NVIDIA_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: NVIDIA_MODEL,
      messages,
      temperature: 0.6,
      top_p: 0.95,
      max_tokens: 2048,
      stream: false
    }),
    signal: AbortSignal.timeout(120000)
  });

  const responseText = await response.text();

  let data;
  try {
    data = responseText ? JSON.parse(responseText) : {};
  } catch {
    data = {};
  }

  if (!response.ok) {
    console.error("NVIDIA API error:", {
      status: response.status,
      message:
        data?.error?.message ||
        data?.message ||
        responseText.slice(0, 1000)
    });

    const error = new Error(
      data?.error?.message ||
      data?.message ||
      `NVIDIA API returned HTTP ${response.status}.`
    );

    error.status = response.status === 429 ? 429 : 502;
    throw error;
  }

  const reply = data?.choices?.[0]?.message?.content;

  if (typeof reply !== "string" || !reply.trim()) {
    const error = new Error("NVIDIA returned an empty response.");
    error.status = 502;
    throw error;
  }

  return reply.trim();
}

// Main AI chat endpoint
app.post("/api/chat", async (req, res) => {
  try {
    const message =
      typeof req.body?.message === "string"
        ? req.body.message.trim()
        : "";

    const messages = normalizeMessages(req.body?.messages);

    if (message) {
      messages.push({
        role: "user",
        content: message.slice(0, 20000)
      });
    }

    if (
      messages.length === 0 ||
      !messages.some((item) => item.role === "user")
    ) {
      return sendError(
        res,
        400,
        "Please provide a message to send to Rairolaxy AI."
      );
    }

    const reply = await getNvidiaReply(messages);

    return res.json({
      success: true,
      reply,
      response: reply,
      provider: "NVIDIA",
      model: NVIDIA_MODEL
    });
  } catch (error) {
    console.error("Chat endpoint error:", {
      message: error.message,
      status: error.status
    });

    return sendError(
      res,
      error.status || 500,
      error.message || "Unable to generate an AI response."
    );
  }
});

// Conversation storage endpoints are paused until PostgreSQL
// is configured again. These routes fail explicitly instead
// of crashing the server or pretending history was saved.

app.post("/api/conversations", (req, res) =>
  databaseUnavailable(res)
);

app.get("/api/conversations", (req, res) =>
  databaseUnavailable(res)
);

app.get("/api/conversations/:id", (req, res) =>
  databaseUnavailable(res)
);

app.post("/api/conversations/:id/messages", (req, res) =>
  databaseUnavailable(res)
);

app.delete("/api/conversations/:id", (req, res) =>
  databaseUnavailable(res)
);

// 404 handler
app.use((req, res) =>
  sendError(res, 404, "API endpoint not found.")
);

// Global error handler
app.use((error, req, res, next) => {
  console.error("Unhandled request error:", {
    message: error.message,
    code: error.code
  });

  if (res.headersSent) return next(error);

  return sendError(res, 500, "Internal server error.");
});

// Start server without waiting for a database.
const server = app.listen(PORT, "0.0.0.0", () => {
  console.log(`Rairolaxy AI backend listening on port ${PORT}`);
  console.log("PostgreSQL temporarily disabled.");
  console.log(`NVIDIA configured: ${Boolean(NVIDIA_API_KEY)}`);
  console.log(`Model: ${NVIDIA_MODEL}`);
});

server.on("error", (error) => {
  console.error("HTTP server error:", {
    message: error.message,
    code: error.code
  });
  process.exit(1);
});

function shutdown(signal) {
  console.log(`${signal} received; shutting down.`);
  server.close(() => process.exit(0));
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
