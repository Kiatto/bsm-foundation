// Serverless proxy: your OpenRouter key stays server-side (Vercel env
// var), never reaches the browser. Edge runtime = native fetch, fast
// cold start.

export const config = { runtime: "edge" };

// Fallback cascade: free-tier models get rate-limited unpredictably.
const MODELS = [
  "nvidia/nemotron-3-ultra-550b-a55b:free",
  "tencent/hy3:free",
  "nvidia/nemotron-3-nano-30b-a3b:free",
  "openai/gpt-oss-20b:free",
];

const MAX_CHARS = 3000;
const PER_MODEL_TIMEOUT_MS = 22000;

const PROMPT = `Extract factual triples from the text below.
Rules:
- Output ONLY a JSON array of [subject, relation, object] triples,
  nothing else. No markdown fences, no commentary.
- relation: short snake_case (e.g. requires, owned_by, born_in).
- subject/object: canonical names as written in the text.
- Extract exhaustively but only facts actually stated in the text.

Text:
`;

// Best-effort per-IP rate limit (resets on cold start — a deterrent,
// not a guarantee; fine for a low-traffic demo).
const hits = new Map();
const WINDOW_MS = 60_000, MAX_PER_WINDOW = 6;

function rateLimited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter(t => now - t < WINDOW_MS);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > MAX_PER_WINDOW;
}

async function callModel(model, text, signal) {
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    signal,
    headers: {
      Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      max_tokens: 3000,
      reasoning: { effort: "low" },
      messages: [{ role: "user", content: PROMPT + text }],
    }),
  });
  if (!res.ok) throw new Error(`upstream ${res.status}`);
  const data = await res.json();
  const msg = data.choices?.[0]?.message || {};
  let txt = msg.content || "";
  if (!txt.includes("[")) txt = msg.reasoning || txt;
  const s = txt.slice(txt.indexOf("["), txt.lastIndexOf("]") + 1);
  const triples = JSON.parse(s).filter(
    (t) => Array.isArray(t) && t.length === 3 &&
           t.every((x) => typeof x === "string" && x.length < 200)
  );
  return { triples, model };
}

export default async function handler(req) {
  if (req.method !== "POST")
    return Response.json({ error: "POST only" }, { status: 405 });

  const ip = req.headers.get("x-forwarded-for")?.split(",")[0] || "anon";
  if (rateLimited(ip))
    return Response.json(
      { error: "rate limit reached — wait a minute and try again" },
      { status: 429 }
    );

  let body;
  try { body = await req.json(); } catch {
    return Response.json({ error: "invalid JSON body" }, { status: 400 });
  }
  const text = (body.text || "").toString().slice(0, MAX_CHARS);
  if (text.trim().length < 10)
    return Response.json({ error: "text too short" }, { status: 400 });
  if (!process.env.OPENROUTER_API_KEY)
    return Response.json(
      { error: "server misconfigured: OPENROUTER_API_KEY not set" },
      { status: 500 }
    );

  const errors = [];
  for (const model of MODELS) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), PER_MODEL_TIMEOUT_MS);
    try {
      const out = await callModel(model, text, ctrl.signal);
      clearTimeout(t);
      if (out.triples.length) return Response.json(out);
      errors.push(`${model}: 0 triples parsed`);
    } catch (e) {
      clearTimeout(t);
      errors.push(`${model}: ${e.message}`);
    }
  }
  return Response.json(
    { error: "all models unavailable right now, try again shortly",
      detail: errors },
    { status: 503 }
  );
}
