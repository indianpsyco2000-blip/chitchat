// api/chat.js
import { HfInference } from '@huggingface/inference';
import { SentenceTransformer } from 'sentence-transformers';
import { IndexFlatL2 } from 'faiss-node';
import fs from 'fs';
import path from 'path';

// === CONFIG ===
const HF_TOKEN = process.env.HF_TOKEN;
const MODEL = 'mistralai/Mistral-7B-Instruct-v0.2';
const EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2';
const CHUNK_SIZE = 500;
const OVERLAP = 50;
const K = 3;

// === LAZY INIT (once per container) ===
let index, chunks, embedder, hf;

async function init() {
  if (index) return;

  const txtPath = path.join(process.cwd(), 'data', 'ashoka_info.txt');
  const text = fs.readFileSync(txtPath, 'utf-8');

  // Chunk
  chunks = chunkText(text, CHUNK_SIZE, OVERLAP);

  // Embed
  embedder = new SentenceTransformer(EMBED_MODEL);
  const embeddings = await embedder.encode(chunks);
  const dim = embeddings[0].length;

  // FAISS
  index = new IndexFlatL2(dim);
  index.add(embeddings);

  // HF Client
  hf = new HfInference(HF_TOKEN);
}

function chunkText(text, size, overlap) {
  const res = [];
  let i = 0;
  while (i < text.length) {
    res.push(text.slice(i, i + size));
    i += size - overlap;
    if (i >= text.length) break;
  }
  return res;
}

const PROMPT = `You are an AI assistant for Ashoka Institute of Technology and Management, Varanasi.
Answer ONLY using the context below. If unrelated, say: "I can only help with Ashoka Institute information."

Context:
{context}

Question: {query}
Answer:`;

// === MAIN HANDLER ===
export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).end();

  const { message } = req.body || {};
  if (!message?.trim()) return res.json({ response: 'Please type a question.' });

  try {
    await init();

    // Retrieve
    const qVec = await embedder.encode([message]);
    const results = index.search(qVec, K);
    const context = results.labels.map(i => chunks[i]).join('\n\n');

    // Generate
    const fullPrompt = PROMPT.replace('{context}', context).replace('{query}', message);
    const out = await hf.textGeneration({
      model: MODEL,
      inputs: `[INST] ${fullPrompt} [/INST]`,
      parameters: { max_new_tokens: 200, temperature: 0.3, return_full_text: false }
    });

    res.json({ response: out.generated_text.trim() });
  } catch (err) {
    console.error(err);
    res.status(500).json({ response: 'Service temporarily unavailable.' });
  }
}

export const config = { api: { bodyParser: true } };