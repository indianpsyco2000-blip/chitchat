# Ashoka Institute AI Chatbot (Cloud Version)

Lightweight RAG chatbot using **Hugging Face + Vercel**.

## Features
- No local model
- Free-tier friendly
- < 5 MB total
- Vercel/Netlify ready

## Deploy

1. Fork this repo
2. Go to [vercel.com](https://vercel.com) → Import
3. Add Environment Variable:
   - `HF_TOKEN` = your Hugging Face token
4. Deploy!

## Local Dev

```bash
cp .env.example .env
npm install
vercel dev