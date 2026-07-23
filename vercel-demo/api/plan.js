// Query planner: turns a natural-language question into a retrieval
// plan over the relation vocabulary actually present in the memory.
// Context-blind by design (never sees the source documents, only the
// question + the relation names) — same discipline as the research
// harness this reuses (examples/pilot_openrouter/planner2.py).

export const config = { maxDuration: 30 };

const MODELS = [
  "google/gemma-4-26b-a4b-it:free",
  "openai/gpt-oss-20b:free",
  "nvidia/nemotron-3-nano-30b-a3b:free",
];

const hits = new Map();
const WINDOW_MS = 60_000, MAX_PER_WINDOW = 20;
function rateLimited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < WINDOW_MS);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > MAX_PER_WINDOW;
}

function prompt(question, rels) {
  return `You are a query planner over a fact memory. You see ONLY the
question and the relation vocabulary below — never any source text.

Relations available (use ONLY these, exact spelling): ${rels.join(", ")}

Output ONLY one JSON object, nothing else, no markdown fences:
{"anchor": "<entity named or implied in the question>",
 "chain": ["<rel1>"] or ["<rel1>", "<rel2>"],
 "constraint": {"relation": "<rel>", "value": "<value>"} or null}

Use 2 hops when the question requires passing through an intermediate
entity before reaching the answer. Pick the closest relation if none
matches perfectly.

Question: ${question}`;
}

async function callModel(model, text, timeoutMs) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      signal: ctrl.signal,
      headers: {
        Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model, max_tokens: 400,
        messages: [{ role: "user", content: text }],
      }),
    });
    if (!res.ok) throw new Error(`upstream ${res.status}`);
    const data = await res.json();
    const msg = data.choices?.[0]?.message || {};
    let txt = msg.content || "";
    if (!txt.includes("{")) txt = msg.reasoning || txt;
    const s = txt.slice(txt.indexOf("{"), txt.lastIndexOf("}") + 1);
    const plan = JSON.parse(s);
    if (!plan.anchor || !Array.isArray(plan.chain))
      throw new Error("malformed plan");
    return plan;
  } finally {
    clearTimeout(t);
  }
}

export default async function handler(req, res) {
  if (req.method !== "POST")
    return res.status(405).json({ error: "POST only" });
  const ip = (req.headers["x-forwarded-for"] || "anon").split(",")[0];
  if (rateLimited(ip))
    return res.status(429).json({ error: "rate limit reached" });

  const { question, relations } = req.body || {};
  if (!question || !Array.isArray(relations) || !relations.length)
    return res.status(400).json({ error: "question and relations[] required" });
  if (!process.env.OPENROUTER_API_KEY)
    return res.status(500).json({ error: "server misconfigured: OPENROUTER_API_KEY not set" });

  const text = prompt(question.slice(0, 500), relations.slice(0, 80));
  const start = Date.now();
  const errors = [];
  for (const model of MODELS) {
    const remaining = 25000 - (Date.now() - start);
    if (remaining < 3000) break;
    try {
      const plan = await callModel(model, text, Math.min(remaining, 12000));
      return res.status(200).json({ plan, model });
    } catch (e) {
      errors.push(`${model}: ${e.message}`);
    }
  }
  return res.status(503).json({ error: "planner unavailable right now", detail: errors });
}
