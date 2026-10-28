
const express = require("express");
const cors = require("cors");
require("dotenv").config();

const app = express();

app.use(cors());
app.use(express.json({ limit: "1mb" }));

const PORT = process.env.PORT || 3000;

// ==========================================
// PROVIDER CONFIGURATION
// ==========================================

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;

const GEMINI_MODEL =
  process.env.GEMINI_MODEL || "gemini-2.5-flash";

const NVIDIA_MODEL =
  process.env.NVIDIA_MODEL ||
  "nvidia/nemotron-3-super-120b-a12b";

const GEMINI_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/" +
  encodeURIComponent(GEMINI_MODEL) +
  ":generateContent";

const NVIDIA_URL =
  "https://integrate.api.nvidia.com/v1/chat/completions";

const GEMINI_TIMEOUT_MS = Math.min(
  Math.max(Number(process.env.GEMINI_TIMEOUT_MS) || 45000, 1000),
  120000
);

const NVIDIA_TIMEOUT_MS = Math.min(
  Math.max(Number(process.env.NVIDIA_TIMEOUT_MS) || 60000, 1000),
  120000
);

const MAX_MESSAGE_CHARS = 20000;
const MAX_HISTORY_MESSAGES = 30;

// ==========================================
// COMMON HELPERS
// ==========================================

function logProviderError(provider, error) {
  // Never print API keys or authorization headers.
  const message = String(error?.message || error)
    .replace(/key=[^&\s]+/gi, "key=[REDACTED]")
    .slice(0, 500);

  console.error(`[${provider}] Request failed: ${message}`);
}

function getMessageText(content) {
  if (typeof content === "string") {
    return content.trim();
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part.text === "string") {
          return part.text;
        }
        return "";
      })
      .join("")
      .trim();
  }

  return "";
}

function normalizeMessages(body = {}) {
  const messages = [];
  const rawMessages = Array.isArray(body.messages)
    ? body.messages.slice(-MAX_HISTORY_MESSAGES)
    : [];

  for (const item of rawMessages) {
    if (!item || typeof item !== "object") continue;

    const role =
      item.role === "assistant"
        ? "assistant"
        : item.role === "user"
          ? "user"
          : null;

    const content = getMessageText(item.content);

    if (!role || !content) continue;

    if (content.length > MAX_MESSAGE_CHARS) {
      throw new Error("A message in the conversation is too long.");
    }

    messages.push({ role, content });
  }

  const latestMessage =
    typeof body.message === "string" ? body.message.trim() : "";

  if (latestMessage) {
    if (latestMessage.length > MAX_MESSAGE_CHARS) {
      throw new Error("Message is too long.");
    }

    const last = messages[messages.length - 1];

    // Avoid duplicate user messages when both fields are supplied.
    if (
      !last ||
      last.role !== "user" ||
      last.content !== latestMessage
    ) {
      messages.push({
        role: "user",
        content: latestMessage
      });
    }
  }

  if (messages.length === 0) {
    throw new Error("Please provide a message.");
  }

  if (messages[messages.length - 1].role !== "user") {
    throw new Error("The latest message must be from the user.");
  }

  return messages;
}

async function fetchJson(url, options, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    timeoutMs
  );

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });

    const rawText = await response.text();

    let data;

    try {
      data = rawText ? JSON.parse(rawText) : {};
    } catch {
      throw new Error(
        `Provider returned invalid JSON (HTTP ${response.status}).`
      );
    }

    if (!response.ok) {
      const message =
        data?.error?.message ||
        data?.message ||
        `HTTP ${response.status}`;

      throw new Error(
        `HTTP ${response.status}: ${String(message).slice(0, 300)}`
      );
    }

    return data;
  } finally {
    clearTimeout(timeout);
  }
}

// ==========================================
// PRIMARY AI: GOOGLE GEMINI
// ==========================================

async function askGemini(messages) {
  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not configured.");
  }

  const contents = [];

  for (const message of messages) {
    const role =
      message.role === "assistant" ? "model" : "user";

    const previous = contents[contents.length - 1];

    // Gemini expects alternating user/model turns.
    if (previous && previous.role === role) {
      previous.parts[0].text += "\n\n" + message.content;
    } else {
      contents.push({
        role,
        parts: [{ text: message.content }]
      });
    }
  }

  // Gemini conversation history must start with a user turn.
  while (contents.length && contents[0].role !== "user") {
    contents.shift();
  }

  if (!contents.length) {
    throw new Error("No valid user message for Gemini.");
  }

  const data = await fetchJson(
    `${GEMINI_URL}?key=${encodeURIComponent(GEMINI_API_KEY)}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        contents,
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 2048
        }
      })
    },
    GEMINI_TIMEOUT_MS
  );

  const parts = data?.candidates?.[0]?.content?.parts || [];

  const answer = parts
    .map((part) =>
      typeof part.text === "string" ? part.text : ""
    )
    .join("")
    .trim();

  if (!answer) {
    const reason = data?.candidates?.[0]?.finishReason;

    throw new Error(
      reason
        ? `Gemini returned no text. Finish reason: ${reason}`
        : "Gemini returned an empty answer."
    );
  }

  return answer;
}

// ==========================================
// FALLBACK AI: NVIDIA
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
        messages,
        temperature: 0.7,
        max_tokens: 2048,
        stream: false
      })
    },
    NVIDIA_TIMEOUT_MS
  );

  const answer = getMessageText(
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
  const geminiConfigured = Boolean(GEMINI_API_KEY);
  const nvidiaConfigured = Boolean(NVIDIA_API_KEY);

  res.json({
    success: true,
    status: "healthy",
    aiConfigured: geminiConfigured || nvidiaConfigured,
    primaryProvider: "Gemini",
    fallbackProvider: "NVIDIA",
    providers: {
      geminiConfigured,
      nvidiaConfigured
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
    aiConfigured: Boolean(GEMINI_API_KEY || NVIDIA_API_KEY),
    primaryProvider: "Gemini",
    fallbackProvider: "NVIDIA",
    fallbackEnabled: Boolean(GEMINI_API_KEY && NVIDIA_API_KEY),
    providers: {
      gemini: {
        configured: Boolean(GEMINI_API_KEY),
        model: GEMINI_MODEL
      },
      nvidia: {
        configured: Boolean(NVIDIA_API_KEY),
        model: NVIDIA_MODEL
      }
    }
  });
});

// ==========================================
// CHAT API: GEMINI FIRST, NVIDIA SECOND
// ==========================================

app.post("/api/chat", async (req, res) => {
  let messages;

  try {
    messages = normalizeMessages(req.body || {});
  } catch (error) {
    return res.status(400).json({
      success: false,
      error: error.message || "Invalid message."
    });
  }

  const failures = [];

  // 1. Try Gemini first.
  if (GEMINI_API_KEY) {
    try {
      const answer = await askGemini(messages);

      return res.json({
        success: true,
        reply: answer,
        answer,
        provider: "Gemini",
        model: GEMINI_MODEL,
        fallbackUsed: false
      });
    } catch (error) {
      logProviderError("Gemini", error);

      failures.push({
        provider: "Gemini",
        error: error.message
      });
    }
  } else {
    failures.push({
      provider: "Gemini",
      error: "GEMINI_API_KEY is not configured."
    });
  }

  // 2. If Gemini fails, try NVIDIA.
  if (NVIDIA_API_KEY) {
    try {
      const answer = await askNvidia(messages);

      return res.json({
        success: true,
        reply: answer,
        answer,
        provider: "NVIDIA",
        model: NVIDIA_MODEL,
        fallbackUsed: true
      });
    } catch (error) {
      logProviderError("NVIDIA", error);

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

  // Both providers failed.
  return res.status(502).json({
    success: false,
    error: "Both AI providers failed to generate a response.",
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
// START SERVER
// ==========================================

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Rairolaxy AI backend listening on port ${PORT}`);
  console.log(`Gemini configured: ${Boolean(GEMINI_API_KEY)}`);
  console.log(`NVIDIA configured: ${Boolean(NVIDIA_API_KEY)}`);
  console.log(`Primary model: ${GEMINI_MODEL}`);
  console.log(`Fallback model: ${NVIDIA_MODEL}`);
});
