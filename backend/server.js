
const express = require("express");
const cors = require("cors");
require("dotenv").config();

const app = express();

app.use(cors());
app.use(express.json({ limit: "12mb" }));

const PORT = process.env.PORT || 3000;

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;

const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const NVIDIA_MODEL =
  process.env.NVIDIA_MODEL ||
  "nvidia/nemotron-3-super-120b-a12b";

const GEMINI_BASE =
  "https://generativelanguage.googleapis.com/v1beta/models";

const NVIDIA_URL =
  "https://integrate.api.nvidia.com/v1/chat/completions";

const GEMINI_TIMEOUT = Math.min(
  Math.max(Number(process.env.GEMINI_TIMEOUT_MS) || 30000, 1000),
  120000
);

const NVIDIA_TIMEOUT = Math.min(
  Math.max(Number(process.env.NVIDIA_TIMEOUT_MS) || 60000, 1000),
  120000
);

const MAX_MESSAGE_CHARS = 20000;
const MAX_HISTORY_MESSAGES = 30;
const MAX_IMAGE_CHARS = 11_200_000;

// ==========================================
// RAIROLAXY AI IDENTITY
// ==========================================

const SYSTEM_PROMPT = `
You are Rairolaxy AI, the official AI assistant created by Rairolaxy.

IDENTITY RULES:
1. Your name and identity are always Rairolaxy AI.
2. Never introduce yourself as Gemini, NVIDIA, Nemotron, or another AI provider.
3. Gemini and NVIDIA are internal AI model providers only.
4. If asked who you are, say:
   "I am Rairolaxy AI, your AI assistant."
5. Never claim that you were created by Google or NVIDIA.
6. Respond naturally in the same language as the user.
7. Be helpful, accurate, clear and conversational.
8. Do not invent facts or pretend to have capabilities you do not have.
9. Do not claim to access a camera or screen unless an image frame
   has actually been supplied by the application.
10. When an image is supplied, analyze only what is visible in it.
11. Never claim to see a continuous live video stream when you
    receive only individual image frames.
12. Keep your application identity separate from your AI provider.

You are Rairolaxy AI. Always maintain this identity.
`;

const VISION_SYSTEM_PROMPT = `
You are Rairolaxy AI, the official AI assistant created by Rairolaxy.

You are analyzing an image frame supplied by the Rairolaxy AI application.

RULES:
1. Your identity is always Rairolaxy AI.
2. Never identify yourself as Gemini, NVIDIA or another provider.
3. Analyze only the supplied image.
4. Describe only things that are actually visible.
5. Do not claim to see outside the supplied image.
6. Do not claim continuous live video access.
7. Answer the user's question directly.
8. Respond in the user's language.
9. If the image is unclear, explain that honestly.
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
// HTTP JSON HELPER
// ==========================================

async function fetchJson(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });

    const text = await response.text();
    let data;

    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      throw new Error(`Invalid JSON response (HTTP ${response.status})`);
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

    if (!["user", "assistant"].includes(item.role)) continue;

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
    .map(model => String(model.name).replace(/^models\//, ""));
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
// GEMINI TEXT CHAT
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

  const parts = data?.candidates?.[0]?.content?.parts || [];

  const answer = parts
    .map(part => typeof part.text === "string" ? part.text : "")
    .join("")
    .trim();

  if (!answer) {
    throw new Error(
      data?.candidates?.[0]?.finishReason
        ? `Gemini returned no text: ${data.candidates[0].finishReason}`
        : "Gemini returned an empty answer."
    );
  }

  return answer;
}

// ==========================================
// GEMINI VISION
// CAMERA + SCREEN ANALYSIS
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

  const parts = data?.candidates?.[0]?.content?.parts || [];

  const answer = parts
    .map(part => typeof part.text === "string" ? part.text : "")
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

  const nvidiaMessages = [
    {
      role: "system",
      content: SYSTEM_PROMPT
    },
    ...messages
  ];

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
        messages: nvidiaMessages,
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
// HEALTH CHECK
// ==========================================

app.get("/health", (req, res) => {
  res.json({
    success: true,
    app: "Rairolaxy AI",
    status: "healthy",
    aiConfigured: Boolean(GEMINI_API_KEY || NVIDIA_API_KEY),
    applicationIdentity: "Rairolaxy AI",
    primaryProvider: "Gemini",
    fallbackProvider: "NVIDIA",
    providers: {
      geminiConfigured: Boolean(GEMINI_API_KEY),
      nvidiaConfigured: Boolean(NVIDIA_API_KEY)
    },
    models: {
      gemini: GEMINI_MODEL,
      nvidia: NVIDIA_MODEL
    }
  });
});

// ==========================================
// PROVIDER STATUS
// ==========================================

app.get("/api/status", (req, res) => {
  res.json({
    success: true,
    app: "Rairolaxy AI",
    applicationIdentity: "Rairolaxy AI",
    aiConfigured: Boolean(GEMINI_API_KEY || NVIDIA_API_KEY),
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
      enabled: Boolean(GEMINI_API_KEY || NVIDIA_API_KEY),
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
      nvidiaConfigured: Boolean(NVIDIA_API_KEY)
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
        console.log(`Rairolaxy AI using Gemini model: ${model}`);

        const answer = await askGemini(messages, model);

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
      console.log("Switching Rairolaxy AI chat to NVIDIA fallback.");

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
// CAMERA / SCREEN IMAGE ANALYSIS API
// POST /api/vision
// Body: { source, image, prompt }
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

  if (typeof image !== "string" || !image) {
    return res.status(400).json({
      success: false,
      error: "Provide an image data URL in the image field."
    });
  }

  if (image.length > MAX_IMAGE_CHARS) {
    return res.status(413).json({
      success: false,
      error: "Image is too large. Reduce resolution or quality."
    });
  }

  if (prompt.length > 4000) {
    return res.status(400).json({
      success: false,
      error: "Vision prompt is too long."
    });
  }

  const candidates = await getGeminiCandidates();
  const failures = [];

  for (const model of candidates) {
    try {
      console.log(
        `Rairolaxy AI vision (${source}) using ${model}`
      );

      const answer = await askGeminiVision(
        image,
        prompt,
        model
      );

      return res.json({
        success: true,
        app: "Rairolaxy AI",
        identity: "Rairolaxy AI",
        answer,
        reply: answer,
        response: answer,
        provider: "Gemini",
        model,
        source
      });
    } catch (error) {
      logError(`Vision ${model}`, error);

      failures.push({
        model,
        error: error.message
      });
    }
  }

  return res.status(502).json({
    success: false,
    app: "Rairolaxy AI",
    error: "Image analysis failed for all available Gemini models.",
    details: failures
  });
});

// ==========================================
// UNKNOWN API ROUTES
// ==========================================

app.use("/api", (req, res) => {
  res.status(404).json({
    success: false,
    error: "API endpoint not found."
  });
});

// ==========================================
// INVALID JSON HANDLER
// ==========================================

app.use((error, req, res, next) => {
  if (error instanceof SyntaxError && "body" in error) {
    return res.status(400).json({
      success: false,
      error: "Invalid JSON request body."
    });
  }

  if (error?.type === "entity.too.large") {
    return res.status(413).json({
      success: false,
      error: "Request body is too large."
    });
  }

  console.error("Unhandled backend error:", error.message);

  return res.status(500).json({
    success: false,
    error: "Internal server error."
  });
});

// ==========================================
// START SERVER
// ==========================================

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Rairolaxy AI backend running on port ${PORT}`);
  console.log(`Gemini configured: ${Boolean(GEMINI_API_KEY)}`);
  console.log(`NVIDIA configured: ${Boolean(NVIDIA_API_KEY)}`);
  console.log(`Vision configured: ${Boolean(GEMINI_API_KEY)}`);
  console.log(`Application identity: Rairolaxy AI`);
  console.log(`Gemini model: ${GEMINI_MODEL}`);
  console.log(`NVIDIA fallback model: ${NVIDIA_MODEL}`);
});
        
