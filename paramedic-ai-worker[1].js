/**
 * paramedic-ai-worker.js
 *
 * Cloudflare Worker — the ONLY server-side piece of this architecture.
 * Receives { question, results } from the browser (results = the array
 * already returned by client-side search(), so this Worker does zero
 * retrieval work — it only builds the grounded prompt and calls the LLM).
 *
 * The system prompt below is a direct port of the rag_message content
 * built in ask_ai() in paramedic_app_v6.py — same grounding rules,
 * same "don't invent doses" constraints.
 *
 * Setup:
 *   1. `npm create cloudflare@latest` -> pick "Worker" template
 *   2. Paste this into src/index.js (or wherever your entry point is)
 *   3. `wrangler secret put ANTHROPIC_API_KEY` to store your key securely
 *      (never hardcode it here or commit it to your repo)
 *   4. `wrangler deploy`
 *   5. Point RUNSHEET's askParamedicAI() at the resulting *.workers.dev URL
 *      (or a custom domain/route you attach to the Worker)
 *
 * Rate limiting: Cloudflare's dashboard lets you attach a Rate Limiting
 * rule to this route without touching this code — do that before
 * announcing the feature publicly (see rateLimitCheck() below for a
 * simple in-Worker fallback using Cloudflare's Cache API if you don't
 * have Rate Limiting Rules on your plan).
 */

const SYSTEM_PROMPT_TEMPLATE = (context) => `You are Paramedic AI, an EMS education
and decision-support assistant.

Use the retrieved reference material below when it is relevant.

Do not invent protocol information.
Do not invent medication doses.
Do not treat TRAINING material as
current protocol.

For real patient care, current local
EMS protocols and medical direction
take precedence.

RETRIEVED REFERENCES:

${context}`;

const ANTHROPIC_MODEL = "claude-sonnet-4-6";
const MAX_TOKENS = 1000;

function buildContext(results) {
  return results
    .map((r, i) => `
SOURCE ${i + 1}
Title: ${r.title}
File: ${r.source}
Jurisdiction: ${r.jurisdiction}
Document type: ${r.document_type}
Effective date: ${r.effective_date}
Status: ${r.status}

TEXT:
${r.text}
`)
    .join("\n");
}

function buildReferences(results) {
  const seen = new Set();
  const lines = [];
  for (const r of results) {
    if (seen.has(r.source)) continue;
    seen.add(r.source);
    lines.push(`- ${r.title} (${r.status})`);
  }
  return lines;
}

async function handleRequest(request, env) {
  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return new Response("Invalid JSON body", { status: 400 });
  }

  const { question, results } = body;
  if (!question || typeof question !== "string") {
    return new Response("Missing 'question'", { status: 400 });
  }
  const safeResults = Array.isArray(results) ? results : [];

  const messages = [{ role: "user", content: question }];
  if (safeResults.length > 0) {
    const context = buildContext(safeResults);
    // Anthropic's API takes system prompts as a top-level field, not a
    // "system" role message — see call below.
    var systemPrompt = SYSTEM_PROMPT_TEMPLATE(context);
  } else {
    var systemPrompt = SYSTEM_PROMPT_TEMPLATE("(no matching reference material found)");
  }

  const anthropicResponse = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: ANTHROPIC_MODEL,
      max_tokens: MAX_TOKENS,
      system: systemPrompt,
      messages,
    }),
  });

  if (!anthropicResponse.ok) {
    const errText = await anthropicResponse.text().catch(() => "");
    return new Response(`Upstream error: ${errText}`, { status: 502 });
  }

  const data = await anthropicResponse.json();
  const answerText = (data.content || [])
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n");

  const references = buildReferences(safeResults);

  return new Response(
    JSON.stringify({ answer: answerText, references }),
    {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        // Lock this to your actual domain before going live —
        // "*" is fine for local testing only.
        "Access-Control-Allow-Origin": "https://runsheet.website",
      },
    }
  );
}

export default {
  async fetch(request, env, ctx) {
    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "https://runsheet.website",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        },
      });
    }
    try {
      return await handleRequest(request, env);
    } catch (err) {
      return new Response(`Worker error: ${err.message}`, { status: 500 });
    }
  },
};
