
const express = require("express");

const router = express.Router();

const BASE_URL = "https://api.inceptionlabs.ai/v1";
const CACHE_MS = 5 * 60 * 1000;

let modelCache = {
  expiresAt: 0,
  chat: [],
  fim: [],
  edit: []
};

function getApiKey() {
  return process.env.INCEPTION_API_KEY;
}

async function inceptionRequest(path, options = {}) {
  const apiKey = getApiKey();

  if (!apiKey) {
    const error = new Error("INCEPTION_API_KEY is not configured");
    error.status = 503;
    throw error;
  }

  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      ...options.headers
    },
    signal: AbortSignal.timeout(120000)
  });

  const data = await response.json().catch(() => ({}));

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
}

async function fetchModelList(path) {
  const result = await inceptionRequest(path);
  return Array.isArray(result.data) ? result.data : [];
}

async function getAllModels(forceRefresh = false) {
  if (!forceRefresh && Date.now() < modelCache.expiresAt) {
    return modelCache;
  }

  const [chat, fim, edit] = await Promise.all([
    fetchModelList("/chat/completions/models"),
    fetchModelList("/fim/completions/models"),
    fetchModelList("/edit/completions/models")
  ]);

  modelCache = {
    expiresAt: Date.now() + CACHE_MS,
    chat,
    fim,
    edit
  };

  return modelCache;
}

// Provider status
router.get("/status", (req, res) => {
  res.json({
    success: true,
    provider: "Inception",
    configured: Boolean(getApiKey())
  });
});

// Get models available to this API account
router.get("/models", async (req, res) => {
  try {
    const models = await getAllModels(
      req.query.refresh === "true"
    );

    res.json({
      success: true,
      provider: "Inception",
      models: {
        chat: models.chat,
        fim: models.fim,
        edit: models.edit
      }
    });
  } catch (error) {
    res.status(error.status || 502).json({
      success: false,
      error: error.message
    });
  }
});

// Chat completion
router.post("/chat", async (req, res) => {
  try {
    const { model, messages, max_tokens, temperature } = req.body;

    if (!model || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        success: false,
        error: "model and messages are required"
      });
    }

    const models = await getAllModels();
    const allowed = models.chat.some(item => item.id === model);

    if (!allowed) {
      return res.status(400).json({
        success: false,
        error: "Selected model is not available for chat on this account"
      });
    }

    const result = await inceptionRequest("/chat/completions", {
      method: "POST",
      body: JSON.stringify({
        model,
        messages,
        max_tokens: max_tokens || 2048,
        ...(temperature !== undefined ? { temperature } : {})
      })
    });

    res.json({
      success: true,
      provider: "Inception",
      model,
      reply: result.choices?.[0]?.message?.content || "",
      usage: result.usage || null,
      raw: result
    });
  } catch (error) {
    res.status(error.status || 502).json({
      success: false,
      provider: "Inception",
      error: error.message
    });
  }
});

// Code fill-in-the-middle
router.post("/fim", async (req, res) => {
  try {
    const { model, ...input } = req.body;

    if (!model) {
      return res.status(400).json({
        success: false,
        error: "model is required"
      });
    }

    const models = await getAllModels();
    if (!models.fim.some(item => item.id === model)) {
      return res.status(400).json({
        success: false,
        error: "Selected model is not available for FIM"
      });
    }

    const result = await inceptionRequest("/fim/completions", {
      method: "POST",
      body: JSON.stringify({ ...input, model })
    });

    res.json({ success: true, provider: "Inception", result });
  } catch (error) {
    res.status(error.status || 502).json({
      success: false,
      error: error.message
    });
  }
});

// Code editing
router.post("/edit", async (req, res) => {
  try {
    const { model, ...input } = req.body;

    if (!model) {
      return res.status(400).json({
        success: false,
        error: "model is required"
      });
    }

    const models = await getAllModels();
    if (!models.edit.some(item => item.id === model)) {
      return res.status(400).json({
        success: false,
        error: "Selected model is not available for editing"
      });
    }

    const result = await inceptionRequest("/edit/completions", {
      method: "POST",
      body: JSON.stringify({ ...input, model })
    });

    res.json({ success: true, provider: "Inception", result });
  } catch (error) {
    res.status(error.status || 502).json({
      success: false,
      error: error.message
    });
  }
});

module.exports = router;
      
