import { HfInference } from '@huggingface/inference';
import fs from 'fs';
import path from 'path';

const HF_TOKEN = process.env.HF_TOKEN;
const MODEL = 'mistralai/Mistral-7B-Instruct-v0.2';
const hf = new HfInference(HF_TOKEN);

// Load and chunk text
let chunks = [];
function loadChunks() {
  if (chunks.length) return;
  const text = fs.readFileSync(path.join(process.cwd(), 'data', 'ashoka_info.txt'), 'utf-8');
  const words = text.split(/\s+/);
  const size = 100; // ~100 words per chunk
  for (let i = 0; i < words.length; i += size - 20) {
    chunks.push(words.slice(i, i + size).join(' '));
  }
}

// Simple BM25 scoring
function bm25(query, doc, k1 = 1.2, b = 0.75) {
  const qTerms = query.toLowerCase().split(/\s+/);
  const dTerms = doc.toLowerCase().split(/\s+/);
  const avgdl = chunks.reduce((a, c) => a + c.split(/\s+/).length, 0) / chunks.length;
  const dl = dTerms.length;
  let score = 0;

  for (const term of qTerms) {
    if (!dTerms.includes(term)) continue;
    const tf = dTerms.filter(t => t === term).length;
    const idf = Math.log(chunks.length / (1 + chunks.filter(c => c.includes(term)).length));
    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)));
  }
  return score;
}

const PROMPT = `You are an AI assistant for Ashoka Institute of Technology and Management, Varanasi.
Answer ONLY using the context below. If unrelated, say: "I can only help with Ashoka Institute information."

Context:
{context}

Question: {query}
Answer:`;

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).end();
  const { message } = req.body || {};
  if (!message?.trim()) return res.json({ response: 'Please type a question.' });

  try {
    loadChunks();

    // Retrieve top 3 chunks using BM25
    const scores = chunks.map((c, i) => ({ text: c, score: bm25(message, c), index: i }));
    const top = scores.sort((a, b) => b.score - a.score).slice(0, 3);
    const context = top.map(t => t.text).join('\n\n');

    // Generate with HF
    const full = PROMPT.replace('{context}', context).replace('{query}', message);
    const out = await hf.textGeneration({
      model: MODEL,
      inputs: `[INST] ${full} [/INST]`,
      parameters: { max_new_tokens: 180, temperature: 0.3 }
    });

    res.json({ response: out.generated_text.trim() });
  } catch (e) {
    console.error(e);
    res.status(500).json({ response: 'Service unavailable.' });
  }
}

export const config = { api: { bodyParser: true } };
