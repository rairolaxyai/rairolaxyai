const express = require("express");
const cors = require("cors");
require("dotenv").config();

const app = express();

app.use(cors());
app.use(express.json({ limit: "12mb" }));

const PORT = process.env.PORT || 3000;

// API KEYS — Render Environment Variables se aayengi
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;
const INCEPTION_API_KEY = process.env.INCEPTION_API_KEY;

// Models
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const NVIDIA_MODEL =
  process.env.NVIDIA_MODEL || "nvidia/nemotron-3-super-120b-a12b";

const GEMINI_BASE =
  "https://generativelanguage.googleapis.com/v1beta/models";
const NVIDIA_URL =
  "https://integrate.api.nvidia.com/v1/chat/completions";
const INCEPTION_BASE = "https://api.inceptionlabs.ai/v1";

const MAX_MESSAGE_CHARS = 20000;
const MAX_HISTORY_MESSAGES = 30;
const MAX_IMAGE_CHARS = 11200000;
const INCEPTION_CACHE_MS = 5 * 60 * 1000;

const GEMINI_TIMEOUT = Math.min(
  Math.max(Number(process.env.GEMINI_TIMEOUT_MS) || 30000, 1000),
  120000
);

const NVIDIA_TIMEOUT = Math.min(
  Math.max(Number(process.env.NVIDIA_TIMEOUT_MS) || 60000, 1000),
  120000
);

const SYSTEM_PROMPT = `
You are Rairolaxy AI, the official AI assistant created by Rairolaxy.

Your identity is always Rairolaxy AI.
Never introduce yourself as Gemini, NVIDIA, Nemotron, Inception,
or another AI provider. These are internal providers only.

If asked who you are, say:
"I am Rairolaxy AI, your AI assistant."

Respond naturally in the same language as the user.
Be helpful, accurate, clear and conversational.
Do not invent facts or claim capabilities you do not have.
Do not claim camera or screen access unless image frames
have actually been supplied by the application.
`;

const VISION_SYSTEM_PROMPT = `
You are Rairolaxy AI, the official AI assistant created by Rairolaxy.

Analyze only the supplied image.
Describe only what is actually visible.
Do not claim to see outside the image.
Do not claim continuous live video access.
Answer directly in the user's language.
If the image is unclear, explain that honestly.
`;

// ==========================================
// ERROR LOGGING
// ==========================================

function logError(provider, error) {
  const message = String(error?.message || error)
    .replace(/key=[^&\s]+/gi, "key=[REDACTED]")
    .replace(/AIza[0-9A-Za-z_-]{20,}/g, "[REDACTED]")
    .slice(0, 400);

  console.error(`[${provider}] ${message}`);
}

// ==========================================
// JSON REQUEST HELPER
// ==========================================

async function fetchJson(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();

  const timer = setTimeout(
    () => controller.abort(),
    timeoutMs
  );

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
      throw new Error(
        `Invalid JSON response (HTTP ${response.status})`
      );
    }

    if (!response.ok) {
      throw new Error(
        `HTTP ${response.status}: ${
          data?.error?.message ||
          data?.message ||
          "Request failed"
        }`
      );
    }

    return data;

  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`Request timed out after ${timeoutMs}ms`);
    }

    throw error;

  } finally {
    clearTimeout(timer);
  }
}

// ==========================================
// MESSAGE HELPERS
// ==========================================

function getText(content) {
  if (typeof content === "string") {
    return content.trim();
  }

  if (Array.isArray(content)) {
    return content
      .map(part =>
        typeof part === "string"
          ? part
          : typeof part?.text === "string"
            ? part.text
            : ""
      )
      .join("")
      .trim();
  }

  return "";
}

function normalizeMessages(body = {}) {
  const result = [];

  const history = Array.isArray(body.messages)
    ? body.messages.slice(-MAX_HISTORY_MESSAGES)
    : [];

  for (const item of history) {
    if (!item || typeof item !== "object") continue;

    if (!["user", "assistant"].includes(item.role)) {
      continue;
    }

    const content = getText(item.content);

    if (!content) continue;

    if (content.length > MAX_MESSAGE_CHARS) {
      throw new Error("A message is too long.");
    }

    result.push({
      role: item.role,
      content
    });
  }

  const message =
    typeof body.message === "string"
      ? body.message.trim()
      : "";

  if (message) {
    if (message.length > MAX_MESSAGE_CHARS) {
      throw new Error("Message is too long.");
    }

    const last = result[result.length - 1];

    if (
      !last ||
      last.role !== "user" ||
      last.content !== message
    ) {
      result.push({
        role: "user",
        content: message
      });
    }
  }

  if (!result.length) {
    throw new Error("Please provide a message.");
  }

  if (result[result.length - 1].role !== "user") {
    throw new Error("The last message must be from the user.");
  }

  return result;
}

// ==========================================
// GEMINI MODEL DISCOVERY
// ==========================================

async function discoverGeminiModels() {
  if (!GEMINI_API_KEY) return [];

  const data = await fetchJson(
    `${GEMINI_BASE}?key=${encodeURIComponent(GEMINI_API_KEY)}`,
    { method: "GET" },
    Math.min(GEMINI_TIMEOUT, 15000)
  );

  return (Array.isArray(data.models) ? data.models : [])
    .filter(model => {
      const name = model?.name || "";
      const methods = model?.supportedGenerationMethods || [];

      return (
        methods.includes("generateContent") &&
        !/embedding|aqa|image|audio|tts|veo|robotics/i.test(name)
      );
    })
    .map(model =>
      String(model.name).replace(/^models\//, "")
    );
}

async function getGeminiCandidates() {
  const models = [];

  function add(model) {
    if (typeof model !== "string") return;

    const name = model.replace(/^models\//, "").trim();

    if (name && !models.includes(name)) {
      models.push(name);
    }
  }

  add(GEMINI_MODEL);

  try {
    const available = await discoverGeminiModels();

    available.sort((a, b) => {
      const aRank = /flash/i.test(a) ? 0 : 1;
      const bRank = /flash/i.test(b) ? 0 : 1;

      return aRank - bRank || a.localeCompare(b);
    });

    available.forEach(add);

  } catch (error) {
    logError("Gemini model discovery", error);
  }

  return models;
}

// ==========================================
// GEMINI CHAT
// ==========================================

async function askGemini(messages, model) {
  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not configured.");
  }

  const contents = [];

  for (const message of messages) {
    const role =
      message.role === "assistant" ? "model" : "user";

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

  if (!contents.length) {
    throw new Error("No valid user message for Gemini.");
  }

  const url =
    `${GEMINI_BASE}/${encodeURIComponent(model)}` +
    `:generateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`;

  const data = await fetchJson(
    url,
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
    GEMINI_TIMEOUT
  );

  const parts =
    data?.candidates?.[0]?.content?.parts || [];

  const answer = parts
    .map(part =>
      typeof part.text === "string" ? part.text : ""
    )
    .join("")
    .trim();

  if (!answer) {
    throw new Error("Gemini returned an empty answer.");
  }

  return answer;
}

// ==========================================
// GEMINI VISION
// CAMERA + SCREEN
// ==========================================

async function askGeminiVision(image, prompt, model) {
  if (!GEMINI_API_KEY) {
    throw new Error("Vision requires GEMINI_API_KEY.");
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
    base64Data.length % 4 !== 0
  ) {
    throw new Error("The image's base64 data is invalid.");
  }

  const url =
    `${GEMINI_BASE}/${encodeURIComponent(model)}` +
    `:generateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`;

  const userPrompt =
    typeof prompt === "string" && prompt.trim()
      ? prompt.trim()
      : "Describe what is visible in this image.";

  const data = await fetchJson(
    url,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        systemInstruction: {
          parts: [{ text: VISION_SYSTEM_PROMPT }]
        },
        contents: [
          {
            role: "user",
            parts: [
              { text: userPrompt },
              {
                inlineData: {
                  mimeType,
                  data: base64Data
                }
              }
            ]
          }
        ],
        generationConfig: {
          temperature: 0.4,
          maxOutputTokens: 1024
        }
      })
    },
    GEMINI_TIMEOUT
  );

  const parts =
    data?.candidates?.[0]?.content?.parts || [];

  const answer = parts
    .map(part =>
      typeof part.text === "string" ? part.text : ""
    )
    .join("")
    .trim();

  if (!answer) {
    throw new Error("Gemini returned an empty vision answer.");
  }

  return answer;
}

// ==========================================
// NVIDIA CHAT FALLBACK
// ==========================================

async function askNvidia(messages) {
  if (!NVIDIA_API_KEY) {
    throw new Error("NVIDIA_API_KEY is not configured.");
  }

  const data = await fetchJson(
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
    NVIDIA_TIMEOUT
  );

  const answer = getText(
    data?.choices?.[0]?.message?.content
  );

  if (!answer) {
    throw new Error("NVIDIA returned an empty answer.");
  }

  return answer;
}

// ==========================================
// INCEPTION PROVIDER
// ==========================================

let inceptionModelCache = {
  expiresAt: 0,
  chat: [],
  fim: [],
  edit: [],
  errors: []
};

async function inceptionRequest(path, options = {}) {
  if (!INCEPTION_API_KEY) {
    const error = new Error(
      "INCEPTION_API_KEY is not configured."
    );
    error.status = 503;
    throw error;
  }

  const controller = new AbortController();

  const timer = setTimeout(
    () => controller.abort(),
    120000
  );

  try {
    const response = await fetch(
      `${INCEPTION_BASE}${path}`,
      {
        ...options,
        headers: {
          Authorization: `Bearer ${INCEPTION_API_KEY}`,
          "Content-Type": "application/json",
          ...options.headers
        },
        signal: controller.signal
      }
    );

    const raw = await response.text();

    let data = {};

    try {
      data = raw ? JSON.parse(raw) : {};
    } catch {
      throw new Error(
        `Inception returned invalid JSON (HTTP ${response.status}).`
      );
    }

    if (!response.ok) {
      const error = new Error(
        data?.error?.message ||
        data?.message ||
        `Inception API returned HTTP ${response.status}`
      );

      error.status = response.status;
      throw error;
    }

    return data;

  } catch (error) {
    if (error.name === "AbortError") {
      const timeoutError = new Error(
        "Inception request timed out."
      );
      timeoutError.status = 504;
      throw timeoutError;
    }

    throw error;

  } finally {
    clearTimeout(timer);
  }
}

async function fetchInceptionModelList(path) {
  const result = await inceptionRequest(path);

  return Array.isArray(result.data)
    ? result.data
    : [];
}

async function getInceptionModels(forceRefresh = false) {
  if (
    !forceRefresh &&
    Date.now() < inceptionModelCache.expiresAt
  ) {
    return inceptionModelCache;
  }

  const paths = [
    "/chat/completions/models",
    "/fim/completions/models",
    "/edit/completions/models"
  ];

  const results = await Promise.allSettled(
    paths.map(path => fetchInceptionModelList(path))
  );

  const errors = results
    .filter(result => result.status === "rejected")
    .map(result =>
      result.reason?.message || "Model lookup failed."
    );

  const models = {
    expiresAt: Date.now() + INCEPTION_CACHE_MS,
    chat: results[0].status === "fulfilled"
      ? results[0].value
      : [],
    fim: results[1].status === "fulfilled"
      ? results[1].value
      : [],
    edit: results[2].status === "fulfilled"
      ? results[2].value
      : [],
    errors
  };

  if (
    !models.chat.length &&
    !models.fim.length &&
    !models.edit.length &&
    errors.length
  ) {
    const error = new Error(errors.join("; "));
    error.status = 502;
    throw error;
  }

  inceptionModelCache = models;

  return models;
}

// ==========================================
// HEALTH CHECK
// ==========================================

app.get("/health", (req, res) => {
  res.json({
    success: true,
    app: "Rairolaxy AI",
    status: "healthy",
    aiConfigured: Boolean(
      GEMINI_API_KEY ||
      NVIDIA_API_KEY ||
      INCEPTION_API_KEY
    ),
    applicationIdentity: "Rairolaxy AI",
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

// ==========================================
// API STATUS
// ==========================================

app.get("/api/status", (req, res) => {
  res.json({
    success: true,
    app: "Rairolaxy AI",
    applicationIdentity: "Rairolaxy AI",
    aiConfigured: Boolean(
      GEMINI_API_KEY ||
      NVIDIA_API_KEY ||
      INCEPTION_API_KEY
    ),
    primaryProvider: "Gemini",
    fallbackProvider: "NVIDIA",
    modelDiscoveryEnabled: true,
    visionConfigured: Boolean(GEMINI_API_KEY),

    providers: {
      gemini: {
        configured: Boolean(GEMINI_API_KEY),
        firstAttemptModel: GEMINI_MODEL
      },
      nvidia: {
        configured: Boolean(NVIDIA_API_KEY),
        model: NVIDIA_MODEL
      },
      inception: {
        configured: Boolean(INCEPTION_API_KEY),
        baseUrl: INCEPTION_BASE
      }
    }
  });
});

// ==========================================
// FRONTEND CAPABILITIES
// ==========================================

app.get("/api/capabilities", (req, res) => {
  res.json({
    success: true,
    app: "Rairolaxy AI",
    identity: "Rairolaxy AI",

    chat: {
      enabled: Boolean(
        GEMINI_API_KEY || NVIDIA_API_KEY
      ),
      endpoint: "/api/chat"
    },

    vision: {
      enabled: Boolean(GEMINI_API_KEY),
      cameraFrameAnalysis: Boolean(GEMINI_API_KEY),
      screenFrameAnalysis: Boolean(GEMINI_API_KEY),
      endpoint: "/api/vision",
      supportedImageTypes: [
        "image/jpeg",
        "image/png",
        "image/webp"
      ],
      continuousLiveStream: false
    },

    providers: {
      geminiConfigured: Boolean(GEMINI_API_KEY),
      nvidiaConfigured: Boolean(NVIDIA_API_KEY),
      inceptionConfigured: Boolean(INCEPTION_API_KEY)
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

// ==========================================
// MAIN CHAT API
// GEMINI PRIMARY -> NVIDIA FALLBACK
// ==========================================

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

  const failures = [];

  if (GEMINI_API_KEY) {
    const candidates = await getGeminiCandidates();

    for (const model of candidates) {
      try {
        console.log(
          `Rairolaxy AI using Gemini model: ${model}`
        );

        const answer = await askGemini(
          messages,
          model
        );

        return res.json({
          success: true,
          app: "Rairolaxy AI",
          reply: answer,
          response: answer,
          answer,
          identity: "Rairolaxy AI",
          provider: "Gemini",
          model,
          fallbackUsed: model !== GEMINI_MODEL
        });

      } catch (error) {
        logError(`Gemini ${model}`, error);

        failures.push({
          provider: "Gemini",
          model,
          error: error.message
        });
      }
    }

    if (!candidates.length) {
      failures.push({
        provider: "Gemini",
        error: "No available Gemini models were discovered."
      });
    }

  } else {
    failures.push({
      provider: "Gemini",
      error: "GEMINI_API_KEY is not configured."
    });
  }

  if (NVIDIA_API_KEY) {
    try {
      console.log(
        "Switching Rairolaxy AI chat to NVIDIA fallback."
      );

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
      logError("NVIDIA fallback", error);

      failures.push({
        provider: "NVIDIA",
        error: error.message
      });
    }

  } else {
    failures.push({
      provider: "NVIDIA",
      error: "NVIDIA_API_KEY is not configured."
    });
  }

  return res.status(502).json({
    success: false,
    app: "Rairolaxy AI",
    error: "All configured AI providers failed.",
    details: failures
  });
});

// ==========================================
// CAMERA / SCREEN VISION API
// POST /api/vision
// ==========================================

app.post("/api/vision", async (req, res) => {
  if (!GEMINI_API_KEY) {
    return res.status(503).json({
      success: false,
      error: "Vision is not configured. Add GEMINI_API_KEY."
    });
  }

  const body = req.body || {};

  const source = body.source;
  const image = body.image;

  const prompt =
    typeof body.prompt === "string"
      ? body.prompt.trim()
      : "";

  if (!["camera", "screen"].includes(source)) {
    return res.status(400).json({
      success: false,
      error: 'source must be either "camera" or "screen".'
    });
  }

  if (typeof ima
