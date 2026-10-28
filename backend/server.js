
const express = require("express");
const cors = require("cors");
require("dotenv").config();

const app = express();

app.use(cors());
app.use(express.json({ limit: "1mb" }));

const PORT = process.env.PORT || 3000;

// AI provider configuration
const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const NVIDIA_MODEL =
  process.env.NVIDIA_MODEL ||
  "nvidia/nemotron-3-super-120b-a12b";

const GEMINI_MODEL =
  process.env.GEMINI_MODEL ||
  "gemini-2.5-flash";

const NVIDIA_URL =
  "https://integrate.api.nvidia.com/v1/chat/completions";

const GEMINI_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/" +
  encodeURIComponent(GEMINI_MODEL) +
  ":generateContent";

const NVIDIA_TIMEOUT_MS = Math.min(
  Math.max(Number(process.env.NVIDIA_TIMEOUT_MS) || 60000, 1000),
  120000
);

const GEMINI_TIMEOUT_MS = Math.min(
  Math.max(Number(process.env.GEMINI_TIMEOUT_MS) || 45000, 1000),
  120000
);

const MAX_MESSAGE_CHARS = 20000;
const MAX_HISTORY_MESSAGES = 30;

// Prevent sensitive provider keys from appearing in logs.
function logProviderError(provider, error) {
  console.error(
    `[${provider}] Request failed:`,
    String(error?.message || error)
      .replace(/key=[^&\s]+/gi, "key=[REDACTED]")
      .slice(0, 500)
  );
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

function normalizeMessages(body) {
  const rawMessages = Array.isArray(body.messages)
    ? body.messages.slice(-MAX_HISTORY_MESSAGES)
    : [];

  const messages = [];

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
    if (content.length > MAX_MESSAGE_CHARS) continue;

    messages.push({ role, content });
  }

  const message = typeof body.message === "string"
    ? body.message.trim()
    : "";

  // Support both { message } and { messages } requests.
  if (message) {
    if (message.length > MAX_MESSAGE_CHARS) {
      throw new Error("Message is too long.");
    }

    const last = messages[messages.length - 1];

    // Avoid duplicating the final user message if both fields contain it.
    if (!(last && last.role === "user" && last.content === message)) {
      messages.push({ role: "user", content: message });
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
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });

    const rawText = await response.text();

    let data = {};

    try {
      data = rawText ? JSON.parse(rawText) : {};
    } catch {
      throw new Error(
        `Provider returned invalid JSON (HTTP ${response.status}).`
      );
    }

    if (!response.ok) {
      const providerMessage =
        data?.error?.message ||
        data?.message ||
        `HTTP ${response.status}`;

      throw new Error(
        `HTTP ${response.status}: ${String(providerMessage).slice(0, 300)}`
      );
    }

    return data;
  } finally {
    clearTimeout(timeout);
  }
}

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

async function askGemini(messages) {
  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not configured.");
  }

  // Gemini uses "user" and "model" roles.
  // Merge consecutive messages with the same role.
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

  // Gemini conversation history should start with a user message.
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

  const answer = (data?.candidates?.[0]?.content?.parts || [])
    .map((part) => typeof part.text === "string" ? part.text : "")
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

// Health check
app.get("/health", (req, res) => {
  res.json({
    success: true,
    status: "healthy",
    aiConfigured: Boolean(NVIDIA_API_KEY || GEMINI_API_KEY),
    providers: {
      nvidiaConfigured: Boolean(NVIDIA_API_KEY),
      geminiConfigured: Boolean(GEMINI_API_KEY)
    },
    primaryProvider: "NVIDIA",
    fallbackProvider: "Gemini",
    models: {
      nvidia: NVIDIA_MODEL,
      gemini: GEMINI_MODEL
    }
  });
});

// AI configuration status
app.get("/api/status", (req, res) => {
  res.json({
    success: true,
    aiConfigured: Boolean(NVIDIA_API_KEY || GEMINI_API_KEY),
    providers: {
      nvidia: {
        configured: Boolean(NVIDIA_API_KEY),
        model: NVIDIA_MODEL
      },
      gemini: {
        configured: Boolean(GEMINI_API_KEY),
        model: GEMINI_MODEL
      }
    },
    fallbackEnabled: Boolean(NVIDIA_API_KEY && GEMINI_API_KEY)
  });
});

// Chat endpoint: NVIDIA first, Gemini fallback.
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

  // Try NVIDIA first.
  if (NVIDIA_API_KEY) {
    try {
      const answer = await askNvidia(messages);

      return res.json({
        success: true,
        reply: answer,
        answer,
        provider: "NVIDIA",
        model: NVIDIA_MODEL,
        fallbackUsed: false
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

  // NVIDIA failed or was unavailable: try Gemini.
  if (GEMINI_API_KEY) {
    try {
      const answer = await askGemini(messages);

      return res.json({
        success: true,
        reply: answer,
        answer,
        provider: "Gemini",
        model: GEMINI_MODEL,
        fallbackUsed: true
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

  // Neither provider produced an answer.
  return res.status(502).json({
    success: false,
    error: "Both AI providers failed to generate a response.",
    details: failures
  });
});

// Unknown API route
app.use("/api", (req, res) => {
  res.status(404).json({
    success: false,
    error: "API endpoint not found."
  });
});

// Start server
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Rairolaxy AI backend listening on port ${PORT}`);
  console.log(`NVIDIA configured: ${Boolean(NVIDIA_API_KEY)}`);
  console.log(`Gemini configured: ${Boolean(GEMINI_API_KEY)}`);
  console.log(`NVIDIA model: ${NVIDIA_MODEL}`);
  console.log(`Gemini model: ${GEMINI_MODEL}`);
});
      
