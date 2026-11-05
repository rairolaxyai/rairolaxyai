
const express = require("express");
const cors = require("cors");
const { Pool } = require("pg");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 10000;

app.use(cors());
app.use(express.json({ limit: "12mb" }));

// ==================================================
// CONFIGURATION
// ==================================================
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY || "";
const INCEPTION_API_KEY = process.env.INCEPTION_API_KEY || "";

const GEMINI_MODEL = (process.env.GEMINI_MODEL || "auto").trim();
const NVIDIA_MODEL =
  process.env.NVIDIA_MODEL || "nvidia/nemotron-3-super-120b-a12b";

const GEMINI_URL =
  "https://generativelanguage.googleapis.com/v1beta/models";
const NVIDIA_URL =
  "https://integrate.api.nvidia.com/v1/chat/completions";
const INCEPTION_URL = "https://api.inceptionlabs.ai/v1";

const GEMINI_CACHE_TIME = 10 * 60 * 1000;

const SYSTEM_PROMPT = `
You are Rairolaxy AI, the official AI assistant created by Rairolaxy.
Your identity is always Rairolaxy AI.
Never introduce yourself as Gemini, NVIDIA, Nemotron or Inception.
These are internal AI providers only.
Respond naturally in the same language as the user.
Be helpful, accurate, clear and conversational.
Never invent facts.
Never claim camera or screen access unless content was supplied.
`;

const VISION_PROMPT = `
You are Rairolaxy AI.
Analyze only the supplied image.
Describe only what is actually visible.
Do not claim continuous live video access.
Answer in the user's language.
`;

// ==================================================
// DATABASE
// ==================================================
let pool = null;
let databaseReady = false;

if (process.env.DATABASE_URL) {
  pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.DATABASE_URL.includes("localhost")
      ? false
      : { rejectUnauthorized: false },
    max: 5,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 10000
  });

  pool.on("error", error => {
    console.error("Database pool error:", safeError(error));
  });
}

async function initializeDatabase() {
  if (!pool) {
    console.log("DATABASE_URL not configured. Database disabled.");
    return;
  }

  try {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS conversations (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        title TEXT DEFAULT 'New Chat',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
      );

      CREATE TABLE IF NOT EXISTS messages (
        id BIGSERIAL PRIMARY KEY,
        conversation_id BIGINT NOT NULL
          REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        provider TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
      );

      CREATE INDEX IF NOT EXISTS idx_conversations_user
        ON conversations(user_id);

      CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages(conversation_id);
    `);

    databaseReady = true;
    console.log("PostgreSQL database connected.");
  } catch (error) {
    databaseReady = false;
    console.error("Database initialization error:", safeError(error));
  }
}

// ==================================================
// HELPERS
// ==================================================
function safeError(error) {
  return String(error?.message || error)
    .replace(/key=[^&\s]+/gi, "key=[REDACTED]")
    .replace(/AIza[0-9A-Za-z_-]{20,}/g, "[REDACTED]")
    .replace(/Bearer\s+[A-Za-z0-9._-]+/gi, "Bearer [REDACTED]")
    .slice(0, 700);
}

async function requestJson(url, options = {}, timeout = 60000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });

    const raw = await response.text();
    let data = {};

    try {
      data = raw ? JSON.parse(raw) : {};
    } catch {
      throw new Error(`Invalid JSON response: HTTP ${response.status}`);
    }

    if (!response.ok) {
      const error = new Error(
        data?.error?.message ||
        data?.message ||
        `API request failed: HTTP ${response.status}`
      );

      error.status = response.status;
      throw error;
    }

    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("AI request timed out.");
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function getText(content) {
  if (typeof content === "string") return content.trim();

  if (Array.isArray(content)) {
    return content.map(item =>
      typeof item === "string"
        ? item
        : typeof item?.text === "string"
          ? item.text
          : ""
    ).join("").trim();
  }

  return "";
}

function normalizeMessages(body = {}) {
  const messages = [];
  const history = Array.isArray(body.messages)
    ? body.messages.slice(-30)
    : [];

  for (const item of history) {
    if (!item || !["user", "assistant"].includes(item.role)) {
      continue;
    }

    const content = getText(item.content);
    if (!content) continue;

    if (content.length > 20000) {
      throw new Error("Message is too long.");
    }

    messages.push({ role: item.role, content });
  }

  const message =
    typeof body.message === "string" ? body.message.trim() : "";

  if (message) {
    if (message.length > 20000) {
      throw new Error("Message is too long.");
    }

    const last = messages[messages.length - 1];

    if (!last || last.role !== "user" || last.content !== message) {
      messages.push({ role: "user", content: message });
    }
  }

  if (!messages.length) {
    throw new Error("Please provide a message.");
  }

  if (messages[messages.length - 1].role !== "user") {
    throw new Error("The last message must be from the user.");
  }

  return messages;
}

function makeGeminiContents(messages) {
  const contents = [];

  for (const message of messages) {
    const role = message.role === "assistant" ? "model" : "user";
    const previous = contents[contents.length - 1];

    if (previous && previous.role === role) {
      previous.parts[0].text += "\n\n" + message.content;
    } else {
      contents.push({
        role,
        parts: [{ text: message.content }]
      });
    }
  }

  while (contents.length && contents[0].role !== "user") {
    contents.shift();
  }

  return contents;
}

// ==================================================
// GEMINI AUTOMATIC MODEL DISCOVERY
// ==================================================
let geminiModelCache = [];
let geminiModelCacheTime = 0;

function isGeneralGeminiModel(modelName) {
  const name = String(modelName || "").toLowerCase();

  const excluded = [
    "embedding",
    "aqa",
    "imagen",
    "veo",
    "tts",
    "audio",
    "speech",
    "transcribe",
    "robotics",
    "computer-use",
    "computer_use",
    "native-audio"
  ];

  if (excluded.some(part => name.includes(part))) {
    return false;
  }

  return name.startsWith("gemini-");
}

function modelPriority(name) {
  const value = name.toLowerCase();

  if (value.includes("flash") && value.includes("lite")) return 2;
  if (value.includes("flash")) return 1;
  if (value.includes("pro")) return 3;

  return 4;
}

async function getAvailableGeminiModels(forceRefresh = false) {
  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not configured.");
  }

  const cacheValid =
    geminiModelCache.length > 0 &&
    Date.now() - geminiModelCacheTime < GEMINI_CACHE_TIME;

  if (!forceRefresh && cacheValid) {
    return geminiModelCache;
  }

  const discovered = [];
  let pageToken = "";

  do {
    const params = new URLSearchParams({
      pageSize: "1000"
    });

    if (pageToken) {
      params.set("pageToken", pageToken);
    }

    const data = await requestJson(
      `${GEMINI_URL}?${params.toString()}&key=${encodeURIComponent(GEMINI_API_KEY)}`,
      { method: "GET" },
      30000
    );

    for (const model of data.models || []) {
      const methods = model.supportedGenerationMethods || [];

      if (
        model.name &&
        methods.includes("generateContent")
      ) {
        const name = model.name.replace(/^models\//, "");

        if (isGeneralGeminiModel(name)) {
          discovered.push(name);
        }
      }
    }

    pageToken = data.nextPageToken || "";
  } while (pageToken);

  const unique = [...new Set(discovered)];

  if (!unique.length) {
    throw new Error(
      "No compatible Gemini chat models found for this API key."
    );
  }

  unique.sort((a, b) => {
    const difference = modelPriority(a) - modelPriority(b);

    if (difference !== 0) return difference;

    return a.localeCompare(b);
  });

  if (
    GEMINI_MODEL &&
    GEMINI_MODEL.toLowerCase() !== "auto" &&
    unique.includes(GEMINI_MODEL)
  ) {
    unique.splice(unique.indexOf(GEMINI_MODEL), 1);
    unique.unshift(GEMINI_MODEL);
  }

  geminiModelCache = unique;
  geminiModelCacheTime = Date.now();

  console.log(
    "Gemini compatible models discovered:",
    unique.length
  );

  return unique;
}

// ==================================================
// GEMINI CHAT
// ==================================================
async function askGemini(messages) {
  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not configured.");
  }

  const models = await getAvailableGeminiModels();
  const contents = makeGeminiContents(messages);
  const failures = [];

  for (const model of models) {
    try {
      console.log("Trying Gemini:", model);

      const data = await requestJson(
        `${GEMINI_URL}/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            systemInstruction: {
              parts: [{ text: SYSTEM_PROMPT }]
            },
            contents,
            generationConfig: {
              temperature: 0.7,
              maxOutputTokens: 2048
            }
          })
        },
        45000
      );

      const answer = getText(
        data?.candidates?.[0]?.content?.parts
      );

      if (!answer) {
        throw new Error("Gemini returned an empty answer.");
      }

      console.log("Gemini success:", model);

      return {
        answer,
        model
      };
    } catch (error) {
      const reason = safeError(error);

      console.error(
        `Gemini model failed (${model}):`,
        reason
      );

      failures.push({ model, error: reason });
    }
  }

  const error = new Error(
    `All ${models.length} discovered Gemini models failed.`
  );

  error.failures = failures;
  throw error;
}

// ==================================================
// NVIDIA FALLBACK
// ==================================================
async function askNvidia(messages) {
  if (!NVIDIA_API_KEY) {
    throw new Error("NVIDIA_API_KEY is not configured.");
  }

  const data = await requestJson(
    NVIDIA_URL,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${NVIDIA_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: NVIDIA_MODEL,
        messages: [
          {
            role: "system",
            content: SYSTEM_PROMPT
          },
          ...messages
        ],
        temperature: 0.7,
        max_tokens: 2048,
        stream: false
      })
    },
    90000
  );

  const answer = getText(
    data?.choices?.[0]?.message?.content
  );

  if (!answer) {
    throw new Error("NVIDIA returned an empty answer.");
  }

  return answer;
}

// ==================================================
// IMAGE DATA VALIDATION
// ==================================================
function parseImageDataUrl(image) {
  if (typeof image !== "string") {
    throw new Error("Image is required.");
  }

  const match = image.match(
    /^data:(image\/(?:jpeg|png|webp));base64,([\s\S]+)$/
  );

  if (!match) {
    throw new Error(
      "Image must be a JPEG, PNG or WEBP base64 data URL."
    );
  }

  const mimeType = match[1];
  const base64Data = match[2].replace(/\s/g, "");

  if (
    !base64Data ||
    !/^[A-Za-z0-9+/]+={0,2}$/.test(base64Data) ||
    base64Data.length % 4 !== 0 ||
    base64Data.length > 11200000
  ) {
    throw new Error("Invalid or oversized image data.");
  }

  return { mimeType, base64Data };
}

// ==================================================
// GEMINI VISION
// CAMERA + SCREEN FRAME ANALYSIS
// ==================================================
async function askGeminiVision(image, prompt) {
  if (!GEMINI_API_KEY) {
    throw new Error("Vision requires GEMINI_API_KEY.");
  }

  const { mimeType, base64Data } =
    parseImageDataUrl(image);

  const models = await getAvailableGeminiModels();
  const failures = [];

  for (const model of models) {
    try {
      console.log("Trying Gemini vision:", model);

      const data = await requestJson(
        `${GEMINI_URL}/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            systemInstruction: {
              parts: [{ text: VISION_PROMPT }]
            },
            contents: [{
              role: "user",
              parts: [
                {
                  text: prompt ||
                    "Describe what is visible in this image."
                },
                {
                  inlineData: {
                    mimeType,
                    data: base64Data
                  }
                }
              ]
            }],
            generationConfig: {
              temperature: 0.4,
              maxOutputTokens: 1024
            }
          })
        },
        45000
      );

      const answer = getText(
        data?.candidates?.[0]?.content?.parts
      );

      if (!answer) {
        throw new Error("Gemini returned an empty vision answer.");
      }

      return { answer, model };
    } catch (error) {
      const reason = safeError(error);

      console.error(
        `Gemini vision failed (${model}):`,
        reason
      );

      failures.push({ model, error: reason });
    }
  }

  const error = new Error(
    "All discovered Gemini models failed to analyze this image."
  );

  error.failures = failures;
  throw error;
}

// ==================================================
// INCEPTION HELPER
// ==================================================
async function inceptionRequest(path, body = null, method = "GET") {
  if (!INCEPTION_API_KEY) {
    const error = new Error("INCEPTION_API_KEY is not configured.");
    error.status = 503;
    throw error;
  }

  const options = {
    method,
    headers: {
      Authorization: `Bearer ${INCEPTION_API_KEY}`,
      "Content-Type": "application/json"
    }
  };

  if (body) {
    options.body = JSON.stringify(body);
  }

  return requestJson(
    `${INCEPTION_URL}${path}`,
    options,
    120000
  );
}

// ==================================================
// HEALTH CHECK
// ==================================================
app.get("/", (req, res) => {
  res.json({
    success: true,
    app: "Rairolaxy AI",
    message: "Rairolaxy AI backend is running."
  });
});

app.get("/health", (req, res) => {
  res.json({
    success: true,
    app: "Rairolaxy AI",
    status: "healthy",
    aiConfigured: Boolean(
      GEMINI_API_KEY || NVIDIA_API_KEY
    ),
    databaseConfigured: Boolean(pool),
    databaseReady,
    primaryProvider: "Gemini",
    fallbackProvider: "NVIDIA",
    providers: {
      geminiConfigured: Boolean(GEMINI_API_KEY),
      nvidiaConfigured: Boolean(NVIDIA_API_KEY),
      inceptionConfigured: Boolean(INCEPTION_API_KEY)
    },
    models: {
      gemini: GEMINI_MODEL,
      nvidia: NVIDIA_MODEL
    }
  });
});

// ==================================================
// API STATUS
// ==================================================
app.get("/api/status", (req, res) => {
  res.json({
    success: true,
    app: "Rairolaxy AI",
    identity: "Rairolaxy AI",
    primaryProvider: "Gemini",
    fallbackProvider: "NVIDIA",
    databaseReady,
    providers: {
      gemini: {
        configured: Boolean(GEMINI_API_KEY),
        model: GEMINI_MODEL
      },
      nvidia: {
        configured: Boolean(NVIDIA_API_KEY),
        model: NVIDIA_MODEL
      },
      inception: {
        configured: Boolean(INCEPTION_API_KEY)
      }
    }
  });
});

// ==================================================
// FRONTEND CAPABILITIES
// ==================================================
app.get("/api/capabilities", (req, res) => {
  res.json({
    success: true,
    app: "Rairolaxy AI",
    chat: {
      enabled: Boolean(
        GEMINI_API_KEY || NVIDIA_API_KEY
      ),
      endpoint: "/api/chat"
    },
    vision: {
      enabled: Boolean(GEMINI_API_KEY),
      endpoint: "/api/vision",
      cameraFrameAnalysis: Boolean(GEMINI_API_KEY),
      screenFrameAnalysis: Boolean(GEMINI_API_KEY),
      continuousLiveStream: false,
      supportedImageTypes: [
        "image/jpeg",
        "image/png",
        "image/webp"
      ]
    },
    database: {
      configured: Boolean(pool),
      ready: databaseReady
    },
    inception: {
      enabled: Boolean(INCEPTION_API_KEY),
      statusEndpoint: "/api/providers/inception/status",
      modelsEndpoint: "/api/providers/inception/models",
      chatEndpoint: "/api/providers/inception/chat",
      fimEndpoint: "/api/providers/inception/fim",
      editEndpoint: "/api/providers/inception/edit"
    }
  });
});

// ==================================================
// GEMINI MODEL DIAGNOSTICS
// GET /api/providers/gemini/models
// ==================================================
app.get("/api/providers/gemini/models", async (req, res) => {
  try {
    const models = await getAvailableGeminiModels(
      req.query.refresh === "true"
    );

    return res.json({
      success: true,
      provider: "Gemini",
      count: models.length,
      models
    });
  } catch (error) {
    return res.status(502).json({
      success: false,
      error: safeError(error)
    });
  }
});

// ==================================================
// MAIN CHAT API
// GEMINI -> NVIDIA FALLBACK
// ==================================================
app.post("/api/chat", async (req, res) => {
  let messages;

  try {
    messages = normalizeMessages(req.body || {});
  } catch (error) {
    return res.status(400).json({
      success: false,
      error: error.message
    });
  }

  const providerErrors = [];

  if (GEMINI_API_KEY) {
    try {
      const result = await askGemini(messages);
      const answer = result.answer;

      return res.json({
        success: true,
        app: "Rairolaxy AI",
        reply: answer,
        response: answer,
        answer,
        identity: "Rairolaxy AI",
        provider: "Gemini",
        model: result.model,
        fallbackUsed: false
      });
    } catch (error) {
      const reason = safeError(error);

      providerErrors.push({
        provider: "Gemini",
        error: reason
      });

      console.error("Gemini all-model failover:", reason);
    }
  }

  if (NVIDIA_API_KEY) {
    try {
      const answer = await askNvidia(messages);

      return res.json({
        success: true,
        app: "Rairolaxy AI",
        reply: answer,
        response: answer,
        answer,
        identity: "Rairolaxy AI",
        provider: "NVIDIA",
        model: NVIDIA_MODEL,
        fallbackUsed: true
      });
    } catch (error) {
      const reason = safeError(error);

      providerErrors.push({
        provider: "NVIDIA",
        error: reason
      });

      console.error("NVIDIA fallback failed:", reason);
    }
  }

  return res.status(502).json({
    success: false,
    error: "All configured AI providers failed.",
    providerErrors
  });
});

// ==================================================
// CAMERA / SCREEN VISION API
// POST /api/vision
// ==================================================
app.post("/api/vision", async (req, res) => {
  const body = req.body || {};

  if (!["camera", "screen"].includes(body.source)) {
    return res.status(400).json({
      success: false,
      error: 'source must be "camera" or "screen".'
    });
  }

  try {
    const result = await askGeminiVision(
      body.image,
      typeof body.prompt === "string"
        ? body.prompt.trim()
        : ""
    );

    return res.json({
      success: true,
      app: "Rairolaxy AI",
      answer: result.answer,
      reply: result.answer,
      response: result.answer,
      provider: "Gemini",
      model: result.model,
      source: body.source
    });
  } catch (error) {
    console.error("Vision error:", safeError(error));

    return res.status(error.status || 502).json({
      success: false,
      error: safeError(error)
    });
  }
});

// ==================================================
// INCEPTION STATUS + MODELS
// ==================================================
app.get("/api/providers/inception/status", (req, res) => {
  res.json({
    success: true,
    configured: Boolean(INCEPTION_API_KEY),
    provider: "Inception",
    baseUrl: INCEPTION_URL
  });
});

app.get("/api/providers/inception/models", async (req, res) => {
  try {
    const [chat, fim, edit] = await Promise.allSettled([
      inceptionRequest("/chat/completions/models"),
      inceptionRequest("/fim/completions/models"),
      inceptionRequest("/edit/completions/models")
    ]);

    return res.json({
      success: true,
      chat: chat.status === "fulfilled"
        ? chat.value.data || []
        : [],
      fim: fim.status === "fulfilled"
        ? fim.value.data || []
        : [],
      edit: edit.status === "fulfilled"
        ? edit.value.data || []
        : [],
      errors: [chat, fim, edit]
        .filter(item => item.status === "rejected")
        .map(item => safeError(item.reason))
    });
  } catch (error) {
    return res.status(error.status || 502).json({
      success: false,
      error: safeError(error)
    });
  }
});

// ==================================================
// INCEPTION CHAT
// ==================================================
app.post("/api/providers/inception/chat", async (req, res) => {
  try {
    const data = await inceptionRequest(
      "/chat/completions",
      req.body,
      "POST"
    );

    return res.json({
      success: true,
      provider: "Inception",
      data
    });
  } catch (error) {
    return res.status(error.status || 502).json({
      success: false,
      error: safeError(error)
    });
  }
});

// ==================================================
// INCEPTION FIM
// ==================================================
app.post("/api/providers/inception/fim", async (req, res) => {
  try {
    const data = await inceptionRequest(
      "/fim/completions",
      req.body,
      "POST"
    );

    return res.json({
      success: true,
      provider: "Inception",
      data
    });
  } catch (error) {
    return res.status(error.status || 502).json({
      success: false,
      error: safeError(error)
    });
  }
});

// ==================================================
// INCEPTION EDIT
// ==================================================
app.post("/api/providers/inception/edit", async (req, res) => {
  try {
    const data = await inceptionRequest(
      "/edit/completions",
      req.body,
      "POST"
    );

    return res.json({
      success: true,
      provider: "Inception",
      data
    });
  } catch (error) {
    return res.status(error.status || 502).json({
      success: false,
      error: safeError(error)
    });
  }
});

// ==================================================
// DATABASE: CREATE CONVERSATION
// ==================================================
app.post("/api/conversations", async (req, res) => {
  if (!databaseReady) {
    return res.status(503).json({
      success: false,
      error: "Database is not ready."
    });
  }

  const userId = String(req.body?.userId || "").trim();
  const title = String(
    req.body?.title || "New Chat"
  ).slice(0, 200);

  if (!userId) {
    return res.status(400).json({
      success: false,
      error: "userId is required."
    });
  }

  try {
    const result = await pool.query(
      `INSERT INTO conversations (user_id, title)
       VALUES ($1, $2) RETURNING *`,
      [userId, title]
    );

    return res.json({
      success: true,
      conversation: result.rows[0]
    });
  } catch (error) {
    console.error("Create conversation error:", safeError(error));

    return res.status(500).json({
      success: false,
      error: "Could not create conversation."
    });
  }
});

// ==================================================
// DATABASE: LIST CONVERSATIONS
// ==================================================
app.get("/api/conversations", async (req, res) => {
  if (!databaseReady) {
    return res.status(503).json({
      success: false,
      error: "Database is not ready."
    });
  }

  const userId = String(req.query.userId || "").trim();

  if (!userId) {
    return res.status(400).json({
      success: false,
      error: "userId is required."
    });
  }

  try {
    const result = await pool.query(
      `SELECT * FROM conversations
       WHERE user_id = $1
       ORDER BY updated_at DESC`,
      [userId]
    );

    return res.json({
      success: true,
      conversations: result.rows
    });
  } catch (error) {
    console.error("List conversations error:", safeError(error));

    return res.status(500).json({
      success: false,
      error: "Could not load conversations."
    });
  }
});

// ==================================================
// DATABASE: SAVE MESSAGE
// ==================================================
app.post("/api/conversations/:id/messages", async (req, res) => {
  if (!databaseReady) {
    return res.status(503).json({
      success: false,
      error: "Database is not ready."
    });
  }

  const conversationId = req.params.id;
  const userId = String(req.body?.userId || "").trim();
  const role = req.body?.role;
  const content = String(req.body?.content || "").trim();
  const provider = req.body?.provider || null;

  if (!userId || !["user", "assistant"].includes(role) || !content) {
    return res.status(400).json({
      success: false,
      error: "userId, valid role and content are required."
    });
  }

  if (content.length > 20000) {
    return res.status(400).json({
      success: false,
      error: "Message is too long."
    });
  }

  try {
    const check = await pool.query(
      `SELECT id FROM conversations
       WHERE id = $1 AND user_id = $2`,
      [conversationId, userId]
    );

    if (!check.rows.length) {
      return res.status(404).json({
        success: false,
        error: "Conversation not found."
      });
    }

    const result = await pool.query(
      `INSERT INTO messages (conversation_id, role, content, provider)
       VALUES ($1, $2, $3, $4)
       RETURNING *`,
      [conversationId, role, content, provider]
    );

    await pool.query(
      `UPDATE conversations
       SET updated_at = NOW()
       WHERE id = $1`,
      [conversationId]
    );

    return res.json({
      success: true,
      message: result.rows[0]
    });
  } catch (error) {
    console.error("Save message error:", safeError(error));

    return res.status(500).json({
      success: false,
      error: "Could not save message."
    });
  }
});

// ==================================================
// DATABASE: GET CONVERSATION MESSAGES
// ==================================================
app.get("/api/conversations/:id/messages", async (req, res) => {
  if (!databaseReady) {
    return res.status(503).json({
      success: false,
      error: "Database is not ready."
    });
  }

  const userId = String(req.query.userId || "").trim();

  if (!userId) {
    return res.status(400).json({
      success: false,
      error: "userId is required."
    });
  }

  try {
    const check = await pool.query(
      `SELECT id FROM conversations
       WHERE id = $1 AND user_id = $2`,
      [req.params.id, userId]
    );

    if (!check.rows.length) {
      return res.status(404).json({
        success: false,
        error: "Conversation not found."
      });
    }

    const result = await pool.query(
      `SELECT * FROM messages
       WHERE conversation_id = $1
       ORDER BY created_at ASC`,
      [req.params.id]
    );

    return res.json({
      success: true,
      messages: result.rows
    });
  } catch (error) {
    console.error("Load messages error:", safeError(error));

    return res.status(500).json({
      success: false,
      error: "Could not load messages."
    });
  }
});

// ==================================================
// SERVER START
// ==================================================
async function startServer() {
  await initializeDatabase();

  app.listen(PORT, "0.0.0.0", () => {
    console.log(
      `Rairolaxy AI backend running on port ${PORT}`
    );
    console.log("Primary AI: Gemini automatic model discovery");
    console.log("Fallback AI: NVIDIA");
    console.log("Vision API: /api/vision");
    console.log("Inception API: enabled when configured");
  });
}

startServer().catch(error => {
  console.error("Server startup error:", safeError(error));
  process.exit(1);
});
      
