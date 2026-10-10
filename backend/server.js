
"use strict";

require("dotenv").config();

const express = require("express");
const cors = require("cors");
const { Pool } = require("pg");
const crypto = require("crypto");

const app = express();

const PORT = Number(process.env.PORT) || 10000;
const DATABASE_URL = process.env.DATABASE_URL;
const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;
const NVIDIA_MODEL =
  process.env.NVIDIA_MODEL || "nvidia/nemotron-3-super-120b-a12b";
const NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions";

app.disable("x-powered-by");
app.use(cors());
app.use(express.json({ limit: "2mb" }));

// --------------------------------------------------
// PostgreSQL connection
// --------------------------------------------------

let pool = null;
let databaseReady = false;

if (DATABASE_URL) {
  pool = new Pool({
    connectionString: DATABASE_URL,
    ssl: { rejectUnauthorized: false },
    max: 5,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 15000
  });

  pool.on("error", (error) => {
    console.error("PostgreSQL pool error:", {
      message: error.message,
      code: error.code
    });
  });
} else {
  console.error(
    "DATABASE_URL is missing. Add it in Render Environment."
  );
}

// --------------------------------------------------
// Database initialization
// --------------------------------------------------

async function initializeDatabase() {
  if (!pool) {
    throw new Error("DATABASE_URL is not configured.");
  }

  const client = await pool.connect();

  try {
    await client.query("SELECT 1");

    await client.query(`
      CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        title TEXT NOT NULL DEFAULT 'New chat',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      );
    `);

    await client.query(`
      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL
          REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL
          CHECK (role IN ('system', 'user', 'assistant')),
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      );
    `);

    await client.query(`
      CREATE INDEX IF NOT EXISTS
      messages_conversation_created_idx
      ON messages (conversation_id, created_at);
    `);

    await client.query(`
      CREATE INDEX IF NOT EXISTS
      conversations_updated_idx
      ON conversations (updated_at DESC);
    `);

    databaseReady = true;
    console.log("PostgreSQL connected; database initialized.");
  } finally {
    client.release();
  }
}

// --------------------------------------------------
// Helpers
// --------------------------------------------------

function makeId() {
  return crypto.randomUUID();
}

function sendError(res, status, message) {
  return res.status(status).json({
    success: false,
    error: message
  });
}

function requireDatabase(res) {
  if (!pool || !databaseReady) {
    sendError(
      res,
      503,
      "Database is not ready. Check the Render deployment logs."
    );
    return false;
  }

  return true;
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

// --------------------------------------------------
// Health and status endpoints
// --------------------------------------------------

app.get("/", (req, res) => {
  res.json({
    success: true,
    name: "Rairolaxy AI Backend",
    status: "running",
    endpoints: [
      "/health",
      "/api/status",
      "/api/chat",
      "/api/conversations"
    ]
  });
});

app.get("/health", async (req, res) => {
  let database = "not_configured";

  if (pool && databaseReady) {
    try {
      await pool.query("SELECT 1");
      database = "connected";
    } catch (error) {
      database = "error";
      console.error("Health check database error:", {
        message: error.message,
        code: error.code
      });
    }
  } else if (pool) {
    database = "initializing";
  }

  const healthy = database === "connected";

  res.status(healthy ? 200 : 503).json({
    success: healthy,
    status: healthy ? "healthy" : "degraded",
    database,
    aiConfigured: Boolean(NVIDIA_API_KEY),
    provider: "NVIDIA",
    model: NVIDIA_MODEL
  });
});

app.get("/api/status", (req, res) => {
  res.json({
    success: true,
    backend: "online",
    databaseConfigured: Boolean(DATABASE_URL),
    databaseReady,
    aiConfigured: Boolean(NVIDIA_API_KEY),
    provider: "NVIDIA",
    model: NVIDIA_MODEL
  });
});

// --------------------------------------------------
// NVIDIA AI
// --------------------------------------------------

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
    console.error("NVIDIA returned an empty response.");
    const error = new Error("NVIDIA returned an empty response.");
    error.status = 502;
    throw error;
  }

  return reply.trim();
}

// POST /api/chat
// Body: { "message": "Hello" }
// Optional: { "messages": [...] }

app.post("/api/chat", async (req, res) => {
  try {
    const message =
      typeof req.body?.message === "string"
        ? req.body.message.trim()
        : "";

    let messages = normalizeMessages(req.body?.messages);

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

// --------------------------------------------------
// Create a conversation
// POST /api/conversations
// Body: { "title": "My chat", "userId": "optional-id" }
// --------------------------------------------------

app.post("/api/conversations", async (req, res) => {
  if (!requireDatabase(res)) return;

  try {
    const id = makeId();

    const title =
      typeof req.body?.title === "string" &&
      req.body.title.trim()
        ? req.body.title.trim().slice(0, 200)
        : "New chat";

    const userId =
      typeof req.body?.userId === "string"
        ? req.body.userId.slice(0, 200)
        : null;

    const result = await pool.query(
      `INSERT INTO conversations (id, user_id, title)
       VALUES ($1, $2, $3)
       RETURNING id, user_id, title, created_at, updated_at`,
      [id, userId, title]
    );

    return res.status(201).json({
      success: true,
      conversation: result.rows[0]
    });
  } catch (error) {
    console.error("Create conversation error:", {
      message: error.message,
      code: error.code
    });

    return sendError(res, 500, "Could not create the conversation.");
  }
});

// --------------------------------------------------
// List conversations
// GET /api/conversations?userId=optional-id
// --------------------------------------------------

app.get("/api/conversations", async (req, res) => {
  if (!requireDatabase(res)) return;

  try {
    const userId =
      typeof req.query.userId === "string"
        ? req.query.userId
        : null;

    let result;

    if (userId) {
      result = await pool.query(
        `SELECT id, user_id, title, created_at, updated_at
         FROM conversations
         WHERE user_id = $1
         ORDER BY updated_at DESC
         LIMIT 100`,
        [userId]
      );
    } else {
      result = await pool.query(
        `SELECT id, user_id, title, created_at, updated_at
         FROM conversations
         ORDER BY updated_at DESC
         LIMIT 100`
      );
    }

    return res.json({
      success: true,
      conversations: result.rows
    });
  } catch (error) {
    console.error("List conversations error:", {
      message: error.message,
      code: error.code
    });

    return sendError(res, 500, "Could not load conversations.");
  }
});

// --------------------------------------------------
// Read one conversation with messages
// GET /api/conversations/:id
// --------------------------------------------------

app.get("/api/conversations/:id", async (req, res) => {
  if (!requireDatabase(res)) return;

  try {
    const conversationResult = await pool.query(
      `SELECT id, user_id, title, created_at, updated_at
       FROM conversations
       WHERE id = $1`,
      [req.params.id]
    );

    if (conversationResult.rowCount === 0) {
      return sendError(res, 404, "Conversation not found.");
    }

    const messagesResult = await pool.query(
      `SELECT id, role, content, created_at
       FROM messages
       WHERE conversation_id = $1
       ORDER BY created_at ASC`,
      [req.params.id]
    );

    return res.json({
      success: true,
      conversation: {
        ...conversationResult.rows[0],
        messages: messagesResult.rows
      }
    });
  } catch (error) {
    console.error("Get conversation error:", {
      message: error.message,
      code: error.code
    });

    return sendError(res, 500, "Could not load the conversation.");
  }
});

// --------------------------------------------------
// Send a message in an existing conversation
// POST /api/conversations/:id/messages
// Body: { "message": "Hello" }
// --------------------------------------------------

app.post("/api/conversations/:id/messages", async (req, res) => {
  if (!requireDatabase(res)) return;

  const conversationId = req.params.id;

  const message =
    typeof req.body?.message === "string"
      ? req.body.message.trim()
      : "";

  if (!message) {
    return sendError(res, 400, "Message cannot be empty.");
  }

  if (message.length > 20000) {
    return sendError(res, 413, "Message is too long.");
  }

  const client = await pool.connect();

  try {
    const conversationResult = await client.query(
      `SELECT id, title
       FROM conversations
       WHERE id = $1`,
      [conversationId]
    );

    if (conversationResult.rowCount === 0) {
      return sendError(res, 404, "Conversation not found.");
    }

    // Load recent conversation context.
    const historyResult = await client.query(
      `SELECT role, content
       FROM messages
       WHERE conversation_id = $1
       ORDER BY created_at DESC
       LIMIT 30`,
      [conversationId]
    );

    const history = historyResult.rows.reverse();

    const userMessageId = makeId();

    await client.query("BEGIN");

    await client.query(
      `INSERT INTO messages
       (id, conversation_id, role, content)
       VALUES ($1, $2, 'user', $3)`,
      [userMessageId, conversationId, message]
    );

    await client.query(
      `UPDATE conversations
       SET updated_at = NOW(),
           title = CASE
             WHEN title = 'New chat' THEN $2
             ELSE title
           END
       WHERE id = $1`,
      [conversationId, message.slice(0, 60)]
    );

    await client.query("COMMIT");

    let reply;

    try {
      reply = await getNvidiaReply([
        {
          role: "system",
          content:
            "You are Rairolaxy AI, a helpful conversational AI assistant. " +
            "Answer the user's question clearly and naturally. " +
            "Reply in the language the user uses unless they request another language."
        },
        ...history,
        {
          role: "user",
          content: message
        }
      ]);
    } catch (error) {
      // The user message is saved even if the AI provider fails.
      throw error;
    }

    const assistantMessageId = makeId();

    await client.query(
      `INSERT INTO messages
       (id, conversation_id, role, content)
       VALUES ($1, $2, 'assistant', $3)`,
      [assistantMessageId, conversationId, reply]
    );

    await client.query(
      `UPDATE conversations
       SET updated_at = NOW()
       WHERE id = $1`,
      [conversationId]
    );

    return res.json({
      success: true,
      conversationId,
      userMessage: {
        id: userMessageId,
        role: "user",
        content: message
      },
      assistantMessage: {
        id: assistantMessageId,
        role: "assistant",
        content: reply
      },
      reply,
      provider: "NVIDIA",
      model: NVIDIA_MODEL
    });
  } catch (error) {
    console.error("Send conversation message error:", {
      message: error.message,
      code: error.code,
      status: error.status,
      detail: error.detail
    });

    try {
      await client.query("ROLLBACK");
    } catch {
      // No active transaction to roll back.
    }

    return sendError(
      res,
      error.status || 500,
      error.message || "Could not send the message."
    );
  } finally {
    client.release();
  }
});

// --------------------------------------------------
// Delete a conversation
// DELETE /api/conversations/:id
// --------------------------------------------------

app.delete("/api/conversations/:id", async (req, res) => {
  if (!requireDatabase(res)) return;

  try {
    const result = await pool.query(
      `DELETE FROM conversations
       WHERE id = $1
       RETURNING id`,
      [req.params.id]
    );

    if (result.rowCount === 0) {
      return sendError(res, 404, "Conversation not found.");
    }

    return res.json({
      success: true,
      deleted: true
    });
  } catch (error) {
    console.error("Delete conversation error:", {
      message: error.message,
      code: error.code
    });

    return sendError(res, 500, "Could not delete the conversation.");
  }
});

// --------------------------------------------------
// 404 handler
// --------------------------------------------------

app.use((req, res) => {
  return sendError(res, 404, "API endpoint not found.");
});

// --------------------------------------------------
// Global error handler
// --------------------------------------------------

app.use((error, req, res, next) => {
  console.error("Unhandled request error:", {
    message: error.message,
    code: error.code
  });

  if (res.headersSent) return next(error);

  return sendError(res, 500, "Internal server error.");
});

// --------------------------------------------------
// Start server
// --------------------------------------------------

async function startServer() {
  try {
    if (!DATABASE_URL) {
      throw new Error(
        "DATABASE_URL is missing in Render Environment."
      );
    }

    if (!NVIDIA_API_KEY) {
      console.warn(
        "NVIDIA_API_KEY is missing. The backend can start, but AI replies will fail until it is configured."
      );
    }

    await initializeDatabase();

    const server = app.listen(PORT, "0.0.0.0", () => {
      console.log(`Rairolaxy AI backend listening on port ${PORT}`);
      console.log(`Database ready: ${databaseReady}`);
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

    const shutdown = async (signal) => {
      console.log(`${signal} received; shutting down.`);

      server.close(async () => {
        try {
          if (pool) await pool.end();
        } catch (error) {
          console.error("Database shutdown error:", error.message);
        }

        process.exit(0);
      });
    };

    process.on("SIGTERM", () => shutdown("SIGTERM"));
    process.on("SIGINT", () => shutdown("SIGINT"));
  } catch (error) {
    console.error("Backend startup failed:", {
      name: error?.name,
      message: error?.message,
      code: error?.code,
      detail: error?.detail,
      hint: error?.hint,
      stack: error?.stack
    });

    process.exit(1);
  }
}

startServer();
        
