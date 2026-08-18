/**
 * paramedic-ai-client.js
 *
 * Browser-side port of your knowledge_chat.py search() function.
 * Mirrors it exactly: same embedding model family (nomic-embed-text),
 * same cosine similarity math, same TOP_K / MIN_SIMILARITY thresholds.
 *
 * Drop this into RUNSHEET as a <script type="module"> or bundle it in
 * with your existing single-file build process.
 *
 * Requires two static JSON files served alongside your site:
 *   - vector_store.json   (same shape as your local rag/vector_store.json:
 *                           a list of { source, text, embedding })
 *   - metadata.json       (same shape as your local knowledge/metadata.json:
 *                           { documents: { <source>: {title, jurisdiction,
 *                           document_type, effective_date, status} } })
 *
 * Uses @huggingface/transformers (the maintained successor to
 * @xenova/transformers) to run nomic-embed-text-v1.5 entirely client-side,
 * via WASM/WebGPU — no server call needed for retrieval.
 *
 * IMPORTANT — read before wiring this up:
 * Nomic's embedding model was trained with task prefixes:
 *   "search_query: "    for the user's question
 *   "search_document: " for the chunks you embedded during ingestion
 * If your ingest_knowledge.py did NOT add "search_document: " before
 * embedding each chunk via Ollama, using "search_query: " here will
 * create an asymmetry that quietly degrades retrieval quality — check
 * ingest_knowledge.py before shipping this. If ingestion embedded raw
 * text with no prefix, drop the QUERY_PREFIX below to match (empty string).
 */

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/+esm";

// ---- Config (mirrors knowledge_chat.py constants) ----
const TOP_K = 3;
const MIN_SIMILARITY = 0.45;
const QUERY_PREFIX = "search_query: "; // set to "" if your ingestion pipeline used no prefix
const VECTOR_STORE_URL = "/data/vector_store.json";
const METADATA_URL = "/data/metadata.json";
const EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5";

// ---- Lazy-loaded singletons (avoid re-downloading/re-initializing) ----
let _extractorPromise = null;
let _vectorsPromise = null;
let _metadataPromise = null;

function getExtractor() {
  if (!_extractorPromise) {
    _extractorPromise = pipeline("feature-extraction", EMBED_MODEL_ID, {
      // quantized 4-bit weights load much faster over the wire; drop
      // to { dtype: "fp32" } if you need max accuracy over speed
      dtype: "q8",
    });
  }
  return _extractorPromise;
}

function getVectors() {
  if (!_vectorsPromise) {
    _vectorsPromise = fetch(VECTOR_STORE_URL).then((r) => {
      if (!r.ok) throw new Error(`Failed to load vector store: ${r.status}`);
      return r.json();
    });
  }
  return _vectorsPromise;
}

function getMetadata() {
  if (!_metadataPromise) {
    _metadataPromise = fetch(METADATA_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load metadata: ${r.status}`);
        return r.json();
      })
      .catch(() => ({ documents: {} }));
  }
  return _metadataPromise;
}

// ---- Cosine similarity — direct port of similarity() in knowledge_chat.py ----
function cosineSimilarity(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  magA = Math.sqrt(magA);
  magB = Math.sqrt(magB);
  if (magA === 0 || magB === 0) return 0;
  return dot / (magA * magB);
}

/**
 * Direct port of search() from knowledge_chat.py.
 * @param {string} question
 * @returns {Promise<Array<{score:number, source:string, title:string,
 *   jurisdiction:string, document_type:string, effective_date:string,
 *   status:string, text:string}>>}
 */
export async function search(question) {
  const [extractor, vectors, metadata] = await Promise.all([
    getExtractor(),
    getVectors(),
    getMetadata(),
  ]);

  if (!vectors || vectors.length === 0) return [];

  const documents = metadata.documents || {};

  const output = await extractor(QUERY_PREFIX + question, {
    pooling: "mean",
    normalize: true,
  });
  const queryEmbedding = Array.from(output.data);

  const results = [];
  for (const item of vectors) {
    const score = cosineSimilarity(queryEmbedding, item.embedding);
    if (score < MIN_SIMILARITY) continue;

    const source = item.source || "UNKNOWN";
    const info = documents[source] || {};

    results.push({
      score: Math.round(score * 10000) / 10000,
      source,
      title: info.title || source,
      jurisdiction: info.jurisdiction || "UNKNOWN",
      document_type: info.document_type || "UNKNOWN",
      effective_date: info.effective_date || "UNKNOWN",
      status: info.status || "UNKNOWN",
      text: item.text || "",
    });
  }

  results.sort((a, b) => b.score - a.score);
  return results.slice(0, TOP_K);
}

/**
 * Full ask flow: retrieve context in-browser, then call your Cloudflare
 * Worker for generation. This is the function RUNSHEET's UI should call.
 *
 * @param {string} question
 * @param {string} workerUrl - your deployed Worker endpoint
 * @returns {Promise<{answer: string, references: Array<object>}>}
 */
export async function askParamedicAI(question, workerUrl) {
  const results = await search(question);

  const response = await fetch(workerUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, results }),
  });

  if (!response.ok) {
    const errText = await response.text().catch(() => "");
    throw new Error(`Worker request failed (${response.status}): ${errText}`);
  }

  return response.json(); // { answer, references }
}
