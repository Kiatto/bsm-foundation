// Query planner v3 — deterministic anchor + classified relation.
//
// WHY THIS SHAPE (measured, 28/07/2026, local Qwen3-4B, same model and
// questions for both):
//     asking the LLM for a JSON retrieval plan   ->  0/3   (0%)
//     asking the LLM to pick a relation by index ->  6/7   (86%, chance 7%)
// The model can map an Italian question onto the right English relation
// name; it cannot reliably emit the plan object. So the plan is now
// ASSEMBLED here instead of generated: the anchor is computed from the
// stored facts (no LLM), and the LLM is left with the one sub-task it
// demonstrably handles — classification.
//
// Three hypotheses were tested and falsified before settling on this
// (see FEEDBACK_LOG.md): passing the entity list to the LLM made it
// worse (0/3 vs 1/3); it was not sampling noise (deterministic across
// repeats at temperature 0 and default); removing the "use 2 hops"
// sentence did not help.
//
// Context-blindness is preserved: the LLM still sees only relation
// names — never the document text, never the entity names.
//
// DECLARED LIMITATION: single-hop only. No observed question so far
// (T1, T2) required a genuine 2-hop chain, so multi-hop is deliberately
// not implemented rather than speculatively built.

import { IS_LOCAL, modelsOrDefault, chatComplete, CALL_TIMEOUT_MS, TOTAL_BUDGET_MS } from "./_llm.js";

export const config = { maxDuration: 60 };

const MODELS = modelsOrDefault([
  "google/gemma-4-26b-a4b-it:free",
  "openai/gpt-oss-20b:free",
  "nvidia/nemotron-3-nano-30b-a3b:free",
]);

const hits = new Map();
const WINDOW_MS = 60_000, MAX_PER_WINDOW = 30;
function rateLimited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < WINDOW_MS);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > MAX_PER_WINDOW;
}

const STOP = new Set(["il","lo","la","i","gli","le","un","uno","una","di","a",
  "da","in","con","su","per","tra","fra","del","della","dei","delle","dello",
  "al","alla","ai","alle","dal","dalla","nel","nella","sul","sulla","e","che",
  "chi","cosa","come","quando","dove","quanto","quanta","quanti","quante",
  "qual","quale","quali","è","e'","sono","ha","hanno","the","of","a","an",
  "is","are","what","which","who","where","when","how","much","many","does",
  "do","for","to","in","on","at","by","with"]);

function tokens(s) {
  return new Set(
    String(s).toLowerCase()
      .replace(/[_\-]/g, " ")
      .replace(/[^\p{L}\p{N} ]/gu, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2 && !STOP.has(w))
  );
}

// The anchor: an entity explicitly named in the question if there is a
// clear match, otherwise the document's dominant subject. Both are
// computed from the stored facts — no model involved, so this step is
// deterministic and free.
function chooseAnchor(question, triples) {
  const subjectCount = new Map();
  const entities = new Set();
  for (const [s, , o] of triples) {
    entities.add(String(s));
    entities.add(String(o));
    subjectCount.set(String(s), (subjectCount.get(String(s)) || 0) + 1);
  }
  const qt = tokens(question);
  let best = null, bestScore = 0;
  for (const e of entities) {
    const et = tokens(e);
    if (!et.size) continue;
    let hit = 0;
    for (const w of et) if (qt.has(w)) hit++;
    const score = hit / et.size;              // fraction of the entity's
    if (score > bestScore) { bestScore = score; best = e; }  // name matched
  }
  if (best && bestScore >= 0.5)
    return { anchor: best, how: "named-in-question", score: +bestScore.toFixed(2) };

  // implicit subject: the entity most facts are about
  let dom = null, domN = 0;
  for (const [s, n] of subjectCount) if (n > domN) { domN = n; dom = s; }
  const total = triples.length || 1;
  return {
    anchor: dom, how: "dominant-subject",
    score: +(domN / total).toFixed(2),
  };
}

function classifyPrompt(question, rels) {
  const numbered = rels.map((r, i) => `${i + 1}. ${r}`).join("\n");
  return `Which ONE of these fields answers the question? Reply with the NUMBER only, nothing else.

${numbered}

Question: ${question}
Number:`;
}

async function pickRelation(model, question, rels, timeoutMs) {
  const txt = await chatComplete(
    model, [{ role: "user", content: classifyPrompt(question, rels) }],
    { maxTokens: 60, timeoutMs }
  );
  const m = txt.match(/\d+/);
  if (!m) throw new Error(`no number in reply: ${txt.slice(0, 40)}`);
  const idx = parseInt(m[0], 10);
  if (!(idx >= 1 && idx <= rels.length))
    throw new Error(`index ${idx} out of range 1..${rels.length}`);
  return rels[idx - 1];
}

export default async function handler(req, res) {
  if (req.method !== "POST")
    return res.status(405).json({ error: "POST only" });
  const ip = (req.headers["x-forwarded-for"] || "anon").split(",")[0];
  if (rateLimited(ip))
    return res.status(429).json({ error: "rate limit reached" });

  const { question, triples, relations } = req.body || {};
  if (!question)
    return res.status(400).json({ error: "question required" });
  const facts = Array.isArray(triples)
    ? triples.filter((t) => Array.isArray(t) && t.length === 3)
    : [];
  if (!facts.length)
    return res.status(400).json({
      error: "triples[] required (the planner computes the anchor from the stored facts)",
    });
  if (!IS_LOCAL && !process.env.OPENROUTER_API_KEY)
    return res.status(500).json({ error: "server misconfigured: OPENROUTER_API_KEY not set" });

  const rels = Array.isArray(relations) && relations.length
    ? relations.slice(0, 80)
    : [...new Set(facts.map((t) => String(t[1])))].sort().slice(0, 80);

  const { anchor, how, score } = chooseAnchor(question, facts);
  if (!anchor)
    return res.status(400).json({ error: "no entities in the provided facts" });

  const start = Date.now();
  const errors = [];
  for (const model of MODELS) {
    const remaining = (IS_LOCAL ? TOTAL_BUDGET_MS : 25000) - (Date.now() - start);
    if (remaining < 3000) break;
    try {
      const rel = await pickRelation(model, question.slice(0, 500), rels,
        Math.min(remaining, IS_LOCAL ? CALL_TIMEOUT_MS : 12000));
      return res.status(200).json({
        plan: { anchor, chain: [rel], constraint: null },
        anchor_method: how, anchor_score: score, model,
      });
    } catch (e) {
      errors.push(`${model}: ${e.message}`);
    }
  }
  return res.status(503).json({ error: "planner unavailable right now", detail: errors });
}
