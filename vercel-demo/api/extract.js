// Serverless proxy: your OpenRouter key stays server-side (Vercel env
// var), never reaches the browser. Node.js runtime (not Edge): Edge
// has a hard ~25s cap that free-tier reasoning models blow past.

import { IS_LOCAL, modelsOrDefault, chatComplete } from "./_llm.js";

export const config = { maxDuration: 60 };

// Fastest/lightest first — free-tier "ultra"/large models (nemotron-
// 550b, hy3) took 60-90s per call in prior testing and are excluded:
// a single one would exceed even this extended budget. Ignored when
// LLM_BASE_URL points at a local server (one model, no cascade).
const MODELS = modelsOrDefault([
  "google/gemma-4-26b-a4b-it:free",
  "openai/gpt-oss-20b:free",
  "nvidia/nemotron-3-nano-30b-a3b:free",
  "nvidia/nemotron-nano-9b-v2:free",
]);

const MAX_CHARS = 3000;
const DEADLINE_MS = 50000; // margin under the 60s function cap

const PROMPT = `Extract factual triples from the text below.
Rules:
- Output ONLY a JSON array of [subject, relation, object] triples,
  nothing else. No markdown fences, no commentary.
- relation: short snake_case (e.g. requires, owned_by, born_in).
- subject/object: normalize to a short snake_case identifier for the
  entity (strip articles like "the"/"a", keep it to 1-4 words) — the
  SAME entity must get the SAME identifier every time it appears.
- One fact per triple: never split a single entity's name across
  multiple triples because of how a sentence is phrased.
- Extract exhaustively but only facts actually stated in the text.

Text:
`;

// Best-effort per-IP rate limit (resets on cold start — a deterrent,
// not a guarantee; fine for a low-traffic demo).
const hits = new Map();
const WINDOW_MS = 60_000, MAX_PER_WINDOW = 20; // one real PDF import = many chunks

function rateLimited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < WINDOW_MS);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > MAX_PER_WINDOW;
}

async function callModel(model, text, timeoutMs) {
  const txt = await chatComplete(
    model, [{ role: "user", content: PROMPT + text }],
    { maxTokens: 1500, timeoutMs }
  );
  const s = txt.slice(txt.indexOf("["), txt.lastIndexOf("]") + 1);
  const triples = JSON.parse(s).filter(
    (tr) => Array.isArray(tr) && tr.length === 3 &&
            tr.every((x) => typeof x === "string" && x.length < 200)
  );
  return { triples, model };
}

export default async function handler(req, res) {
  if (req.method !== "POST")
    return res.status(405).json({ error: "POST only" });

  const ip = (req.headers["x-forwarded-for"] || "anon").split(",")[0];
  if (rateLimited(ip))
    return res.status(429).json({
      error: "rate limit reached — wait a minute and try again",
    });

  const text = (req.body?.text || "").toString().slice(0, MAX_CHARS);
  if (text.trim().length < 10)
    return res.status(400).json({ error: "text too short" });
  if (!IS_LOCAL && !process.env.OPENROUTER_API_KEY)
    return res.status(500).json({
      error: "server misconfigured: OPENROUTER_API_KEY not set",
    });

  const start = Date.now();
  const errors = [];
  for (const model of MODELS) {
    const remaining = DEADLINE_MS - (Date.now() - start);
    if (remaining < 4000) break;
    try {
      const out = await callModel(model, text, Math.min(remaining, 18000));
      if (out.triples.length) return res.status(200).json(out);
      errors.push(`${model}: 0 triples parsed`);
    } catch (e) {
      errors.push(`${model}: ${e.message}`);
    }
  }
  return res.status(503).json({
    error: "all models unavailable or too slow right now, try again shortly",
    detail: errors,
  });
}
