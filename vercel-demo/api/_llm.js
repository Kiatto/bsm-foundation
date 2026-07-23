// Shared LLM endpoint config: OpenRouter by default, or your local
// Unsloth server (see ../local-llm-server/server.py) via env vars.
//
//   LLM_BASE_URL   default https://openrouter.ai/api/v1/chat/completions
//   LLM_API_KEY    default process.env.OPENROUTER_API_KEY
//   LLM_MODELS     comma-separated override for the model fallback list
//                  (ignored by the local server, which only has one
//                  model loaded — but harmless to pass)

export const BASE_URL = process.env.LLM_BASE_URL
  || "https://openrouter.ai/api/v1/chat/completions";
export const API_KEY = process.env.LLM_API_KEY || process.env.OPENROUTER_API_KEY;
export const IS_LOCAL = BASE_URL.includes("localhost")
  || BASE_URL.includes("127.0.0.1");

export function modelsOrDefault(defaults) {
  if (process.env.LLM_MODELS) return process.env.LLM_MODELS.split(",");
  if (IS_LOCAL) return ["local"]; // one call, no fallback cascade needed
  return defaults;
}

export async function chatComplete(model, messages, { maxTokens = 1500, timeoutMs = 18000 } = {}) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(BASE_URL, {
      method: "POST",
      signal: ctrl.signal,
      headers: {
        Authorization: `Bearer ${API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model, max_tokens: maxTokens, messages }),
    });
    if (!res.ok) throw new Error(`upstream ${res.status}`);
    const data = await res.json();
    const msg = data.choices?.[0]?.message || {};
    let txt = msg.content || "";
    if (!txt.includes("[") && !txt.includes("{")) txt = msg.reasoning || txt;
    return txt;
  } finally {
    clearTimeout(t);
  }
}
