# 🩺 Health AI — Personal Medical Assistant

A fully offline, privacy-focused medical AI system. Your health documents live on your device, retrieval happens locally, and only the selected context reaches the LLM for generation. No cloud, no subscriptions, no data leaving your network.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Backend Setup — aichatbot_v2_fixed](#backend-setup)
- [Frontend — Choose Your UI](#frontend--choose-your-ui)
  - [Option A — Use the Provided App (HealthAIChat)](#option-a--use-the-provided-app-healthaichat)
  - [Option B — Bring Your Own UI](#option-b--bring-your-own-ui)
- [Running the Full System](#running-the-full-system)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Supported Document Types](#supported-document-types)
- [Troubleshooting](#troubleshooting)

---

## Overview

Health AI lets you chat with your own medical records — lab reports, prescriptions, medical history — using a local LLM. The system is split into two parts:

| Part | What it does | Where it runs |
|---|---|---|
| `aichatbot_v2_fixed` | FastAPI server — OCR, embedding, RAG, LLM | Your laptop |
| `HealthAIChat` | React Native app — chat UI, local vector store, on-device retrieval | Your phone |

Both must be on the **same WiFi network**. The phone never connects to the internet — all traffic stays local.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      PHONE (iOS / Android)               │
│                                                          │
│   ┌──────────────────────────────────────────────────┐  │
│   │  HealthAIChat (React Native / Expo)              │  │
│   │                                                  │  │
│   │  • Chat UI                                       │  │
│   │  • Document upload                               │  │
│   │  • Local chunk store (AsyncStorage)              │  │
│   │  • On-device cosine similarity (RAG retrieval)   │  │
│   └──────────────┬───────────────────────────────────┘  │
│                  │  HTTP (same WiFi LAN)                 │
└──────────────────┼──────────────────────────────────────┘
                   │
┌──────────────────┼──────────────────────────────────────┐
│                  ▼          LAPTOP                       │
│   ┌──────────────────────────────────────────────────┐  │
│   │  aichatbot_v2_fixed (FastAPI + Python)           │  │
│   │                                                  │  │
│   │  ┌────────────┐  ┌──────────────┐  ┌─────────┐  │  │
│   │  │  OCR       │  │  Embedder    │  │  FAISS  │  │  │
│   │  │  PaddleOCR │  │  MiniLM-L6   │  │  Index  │  │  │
│   │  │  Tesseract │  │  (384-dim)   │  │         │  │  │
│   │  └────────────┘  └──────────────┘  └─────────┘  │  │
│   │                                                  │  │
│   │  ┌──────────────────────────────────────────┐   │  │
│   │  │  LLM — Meta-Llama-3.1-8B-Instruct        │   │  │
│   │  │  Quantization: Q4_K_M (GGUF)             │   │  │
│   │  │  Runtime: llama-cpp-python               │   │  │
│   │  └──────────────────────────────────────────┘   │  │
│   └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### On-Device RAG Flow

Once you upload a document, this is what happens for every query:

```
1. Phone → POST /embed-query         → Server returns 384-dim query vector
2. Phone runs cosine similarity       → Picks top-5 most relevant chunks locally
3. Phone → POST /generate             → Server runs LLM with those 5 chunks
4. Response displayed in chat
```

The **full document text never leaves the phone** after initial upload. Only the matched chunk snippets are sent to the LLM.

---

## Project Structure

```
aichatbot_v2_fixed/
├── health_ai/
│   ├── api/
│   │   ├── main.py                  ← Entry point (run this)
│   │   ├── server.py                ← FastAPI app + all endpoints
│   │   └── rag_pipeline.py          ← Query classifier + RAG pipeline
│   ├── config/
│   │   └── settings.py              ← Paths, token budgets, model path
│   ├── core/
│   │   ├── clinical_analyzer.py     ← Classifies lab values as H/L/Normal
│   │   ├── profile_manager.py       ← Per-profile directory management
│   │   └── logger.py
│   ├── data/
│   │   └── profiles/
│   │       └── profile-1/
│   │           ├── raw_documents/
│   │           │   ├── blood_reports/    ← PDF lab reports go here
│   │           │   └── prescriptions/   ← JPG/PNG prescriptions go here
│   │           ├── vector_store/         ← FAISS index (auto-created)
│   │           └── processed_files.json  ← Tracks what's been indexed
│   ├── embeddings/
│   │   └── embedder.py              ← SentenceTransformer (all-MiniLM-L6-v2)
│   ├── ingestion/
│   │   ├── ingest.py                ← Main ingestion engine
│   │   ├── startup_indexer.py       ← Auto-indexes raw_documents on startup
│   │   ├── lab_parser.py            ← Generic lab report parser
│   │   └── sterling_accuris_parser.py ← Sterling Accuris specific parser
│   ├── model/
│   │   ├── llm_loader.py            ← Llama wrapper (singleton)
│   │   └── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  ← ⬅ MODEL FILE GOES HERE
│   ├── rag/
│   │   ├── chunker.py               ← Splits text into 500-word chunks
│   │   ├── retriever.py             ← FAISS similarity search
│   │   ├── vector_store.py          ← FAISS index wrapper
│   │   └── context_builder.py       ← Assembles prompt context
│   ├── safety/
│   │   ├── red_flag.py              ← Detects urgent medical keywords
│   │   └── disclaimer.py            ← Appended to every response
│   └── utils/
│       ├── document_reader.py       ← PDF + image text extraction
│       └── prescription_cleaner.py  ← Cleans OCR noise

HealthAIChat/
├── App.js                           ← Entire React Native app (single file)
├── app.json                         ← Expo config (usesCleartextTraffic: true)
├── package.json
└── assets/
```

---

## Prerequisites

### Laptop

| Requirement | Version |
|---|---|
| Python | 3.10 or 3.12 |
| pip | latest |
| RAM | 8 GB minimum, 16 GB recommended |
| Storage | ~6 GB (model file ~4.9 GB + dependencies) |

### Phone

| Requirement | Details |
|---|---|
| OS | iOS 16+ or Android 10+ |
| App | Expo Go (from App Store / Play Store) |
| Network | Same WiFi as laptop |

---

## Backend Setup

### 1. Download the model

Download the GGUF model file from Hugging Face:

**Model:** `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`  
**Source:** https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF  
**Size:** ~4.9 GB

Place it here — the filename must match exactly:

```
aichatbot_v2_fixed/
  health_ai/
    model/
      Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf   ← here
```

If you use a different quantization, update the path in `health_ai/config/settings.py`:

```python
LLM_MODEL_PATH = BASE_DIR / "model" / "your-filename.gguf"
```

### 2. Install Python dependencies

```bash
cd aichatbot_v2_fixed
pip install -r requirements.txt
```

Key packages installed:

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | API server |
| `llama-cpp-python` | Runs the GGUF model |
| `sentence-transformers` | Embedding model (all-MiniLM-L6-v2) |
| `faiss-cpu` | Vector similarity search |
| `paddleocr` | Primary OCR for images |
| `pytesseract` | Fallback OCR |
| `pdfplumber` | PDF text extraction |

### 3. (Optional) Pre-load documents

Drop files into the appropriate folders before starting the server — they'll be auto-indexed on startup:

```
health_ai/data/profiles/profile-1/raw_documents/
  blood_reports/      ← PDF lab reports
  prescriptions/      ← JPG or PNG prescription images
```

### 4. Start the server

```bash
cd aichatbot_v2_fixed
python -m health_ai.api.main
```

On startup you'll see:

```
═══════════════════════════════════════════════════════
  Health AI Backend starting up
═══════════════════════════════════════════════════════
  Local:   http://localhost:8000
  Network: http://192.168.x.x:8000   ← use this on the phone
═══════════════════════════════════════════════════════
```

Write down the **Network** IP — you'll need it in the app.

#### Optional startup flags

```bash
# Use a different port
python -m health_ai.api.main --port 9000

# Force re-OCR all prescription images (useful after OCR upgrades)
python -m health_ai.api.main --force-reindex-images

# Enable auto-reload on code changes (development only)
python -m health_ai.api.main --reload
```

---

## Frontend — Choose Your UI

The backend is a plain HTTP API. The frontend is completely separate — **you have two options** depending on your situation.

---

### Option A — Use the Provided App (HealthAIChat)

This is the React Native app included in this repository. It was built specifically for this backend and supports on-device RAG, document upload, and a full chat interface. Use this option if you don't have your own UI or just want to get started quickly.

#### What you need

| Requirement | Details |
|---|---|
| Node.js | 18 or higher |
| Expo Go app | iOS: App Store · Android: Play Store (get the latest version) |
| Network | Phone and laptop on the same WiFi |

#### Step 1 — Install dependencies

```bash
cd HealthAIChat
npm install --legacy-peer-deps
```

#### Step 2 — Start the Expo dev server

```bash
npx expo start
```

A QR code appears in the terminal.

#### Step 3 — Open on your phone

- **iOS:** Open the Camera app and scan the QR code, or open Expo Go → Scan
- **Android:** Open Expo Go → tap "Scan QR Code"

#### Step 4 — Connect to the server

When the app first opens, a settings dialog appears automatically:

1. Enter the **Network IP** printed at server startup — e.g. `192.168.1.42`
2. Leave port as `8000` (change only if you used `--port`)
3. Tap **Test Connection** → you should see ✅ Server reachable!
4. Tap **Save**

The IP is saved on the phone and remembered across restarts. You can change it anytime by tapping ⚙️ in the top right.

#### What the app does

- **Chat tab** — send queries, get AI responses with rich formatting
- **Documents tab** — upload PDFs and prescription images; they get OCR'd on the server and the embeddings are stored locally on the phone for on-device retrieval
- On-device RAG: the phone runs cosine similarity itself and only sends the top matching chunks to the server for LLM generation — your full document text never leaves the device after upload

---

### Option B — Bring Your Own UI

The server exposes a standard REST API over HTTP. If you already have your own frontend — a web app, a mobile app, an APK, or anything else — you can integrate it directly without using HealthAIChat at all.

#### The only thing your UI needs to know

When the server starts it prints its LAN address:

```
  Network: http://192.168.x.x:8000   ← point your UI at this
```

Your UI needs to let the user enter this IP and port somewhere — a settings screen, a config file, an input field on first launch, anything. Once it has the base URL, all endpoints work the same way.

#### Minimum integration — simple chat

If all you want is a basic chat against the user's documents, you only need one endpoint:

```
POST http://<server-ip>:<port>/query/profile-1
Content-Type: application/json

{
  "query": "What are my abnormal lab results?",
  "history": ["previous user message", "previous AI reply"]   ← optional, pass [] to start fresh
}
```

Response:

```json
{
  "response": "Your Hemoglobin is 10.2 g/dL which is below the normal range..."
}
```

That's it. The server handles OCR, chunking, embedding, retrieval, and LLM generation entirely on its own. Your UI just sends a query and displays the response.

#### Full on-device RAG integration (recommended for mobile)

If you want retrieval to happen on the device rather than the server — better for privacy and works even when the server is slow — use these three endpoints in sequence:

**1. Upload a document and get embeddings back**

```
POST http://<server-ip>:<port>/upload-and-embed/profile-1
Content-Type: multipart/form-data

file: <PDF or image file>
```

Response includes `chunks` — an array of `{ text, embedding, metadata }` objects. Store these locally on the device.

**2. On user query — embed the query**

```
POST http://<server-ip>:<port>/embed-query
Content-Type: application/json

{ "query": "What medicines am I taking?" }
```

Response: `{ "embedding": [0.023, -0.041, ...] }` — a 384-dimensional float array.

**3. Run cosine similarity on device, then generate**

Compute dot product between the query embedding and each stored chunk embedding (vectors are pre-normalised so dot product = cosine similarity). Pick the top 5. Then:

```
POST http://<server-ip>:<port>/generate/profile-1
Content-Type: application/json

{
  "query": "What medicines am I taking?",
  "chunks": ["chunk text 1", "chunk text 2", "chunk text 3", "chunk text 4", "chunk text 5"],
  "history": []
}
```

Response: `{ "response": "..." }`

#### Cosine similarity — code snippets

If you need a reference implementation for the on-device similarity calculation:

**JavaScript / React Native**
```javascript
function cosineSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // vectors are L2-normalised, dot product == cosine similarity
}

function topK(queryEmbedding, chunks, k = 5) {
  return chunks
    .map(c => ({ text: c.text, score: cosineSim(queryEmbedding, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map(c => c.text);
}
```

**Python**
```python
import numpy as np

def top_k(query_embedding, chunks, k=5):
    query = np.array(query_embedding)
    scores = [(c["text"], float(np.dot(query, c["embedding"]))) for c in chunks]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in scores[:k]]
```

**Kotlin (Android)**
```kotlin
fun cosineSim(a: FloatArray, b: FloatArray): Float {
    var dot = 0f
    for (i in a.indices) dot += a[i] * b[i]
    return dot
}

fun topK(queryEmbedding: FloatArray, chunks: List<Chunk>, k: Int = 5): List<String> {
    return chunks
        .sortedByDescending { cosineSim(queryEmbedding, it.embedding) }
        .take(k)
        .map { it.text }
}
```

#### Health check — verify the server is reachable

Before any other call, ping this to confirm connectivity:

```
GET http://<server-ip>:<port>/health
```

Returns `{ "status": "ok" }` if the server is up. Use this to show a connected/disconnected indicator in your UI.

---

## Running the Full System

### If using HealthAIChat (Option A)

Open two terminals on your laptop:

```
Terminal 1                          Terminal 2
──────────────────────────────      ──────────────────────────────
cd aichatbot_v2_fixed               cd HealthAIChat
python -m health_ai.api.main        npx expo start
                                    (scan QR code with phone)
```

Once the phone has loaded the app through Expo Go, Terminal 2 can be closed. Terminal 1 must stay open the entire time you are using the app.

### If using your own UI (Option B)

You only need one terminal:

```bash
cd aichatbot_v2_fixed
python -m health_ai.api.main
```

Then open your UI, enter the IP and port shown in the terminal output, and you're connected. No second terminal needed.

---

## How It Works

### Query classification

Every query is automatically classified before processing:

| Intent | Trigger words | Handling |
|---|---|---|
| `general` | No personal keywords | LLM answers from training knowledge |
| `personal` | "my", "report", "result", "blood" | RAG over indexed documents |
| `prescription` | "prescription", "medicine", "tablet", "dosage" | Dedicated prescription retrieval |
| `symptom` | "I feel", "pain", "fever", "headache" | Symptom guidance prompt |
| `summary` | "summary", "all results", "everything" | Deterministic Python summary (no LLM) |

### OCR pipeline for prescriptions

Images go through two OCR engines in order:

1. **PaddleOCR** (primary) — confidence threshold 0.40, keeps handwritten text
2. **pytesseract** (fallback) — PSM 6 layout for prescription formats

### Document indexing

Every uploaded document produces two types of stored data:

1. **Server FAISS index** — used when no local chunks exist (fallback RAG)
2. **Phone AsyncStorage** — chunk text + 384-dim embeddings, used for on-device retrieval

### Chunking

Documents are split into 500-word chunks with 100-word overlap to preserve context across boundaries.

---

## API Reference

All endpoints are on `http://<laptop-ip>:8000`.

### Core

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server health check |
| GET | `/server/info` | IP, port, vector count |
| GET | `/profiles/{profile_id}/status` | Vectors indexed for a profile |

### On-Device RAG (used by the app)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload-and-embed/{profile_id}` | Upload file → get chunks + embeddings back |
| POST | `/embed-query` | Embed a query string → get vector back |
| POST | `/generate/{profile_id}` | LLM generation with phone-supplied chunks |

### Server-Side RAG (fallback)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/query/{profile_id}` | Full server-side RAG + LLM |
| POST | `/upload/{profile_id}` | Upload file, index server-side only |

### Structured data ingestion

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ingest/medical-history/{profile_id}` | Ingest medical history JSON |
| POST | `/ingest/prescriptions/{profile_id}` | Ingest prescription JSON |
| POST | `/ingest/hospitalizations/{profile_id}` | Ingest hospitalization records |
| POST | `/ingest/family-history/{profile_id}` | Ingest family history |
| POST | `/ingest/daily-log/{profile_id}` | Ingest daily health log |

### Example query

```bash
curl -X POST http://192.168.1.42:8000/query/profile-1 \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my abnormal lab results?", "history": []}'
```

---

## Supported Document Types

| Type | Formats | Notes |
|---|---|---|
| Lab reports | PDF | Sterling Accuris format has a dedicated parser; other formats use generic table/regex parser |
| Prescriptions | JPG, JPEG, PNG | OCR'd with PaddleOCR then pytesseract fallback |
| Medical history | JSON (via API) | Structured schema — use `/ingest/medical-history` |
| Hospitalizations | JSON (via API) | Use `/ingest/hospitalizations` |
| Family history | JSON (via API) | Use `/ingest/family-history` |

---

## Troubleshooting

### Android — "Network request failed" or no connection

Android 9+ blocks plain HTTP by default. Make sure `app.json` contains:

```json
"android": {
  "usesCleartextTraffic": true
}
```

If you edited `app.json`, restart Expo completely (`Ctrl+C` then `npx expo start`).

---

### iOS — "Project is incompatible with this version of Expo Go"

The Expo SDK version in `package.json` must match the Expo Go version installed on your phone.

| Expo Go version | Required SDK in package.json |
|---|---|
| 54.x | `"expo": "~54.0.0"` |
| 53.x | `"expo": "~53.0.0"` |

After editing `package.json`, run:

```bash
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
npx expo start
```

---

### Server won't start — model not found

Check that the model file path matches exactly:

```
aichatbot_v2_fixed/health_ai/model/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

If you downloaded a different quantization, edit `health_ai/config/settings.py`:

```python
LLM_MODEL_PATH = BASE_DIR / "model" / "your-actual-filename.gguf"
```

---

### OCR returns empty text for prescription images

Try re-indexing with the force flag:

```bash
python -m health_ai.api.main --force-reindex-images
```

If still empty, ensure the image is:
- In focus and well-lit
- At least 300 DPI
- Under 10 MB in file size

---

### Prescription chunks not found in chat

The registry tracks which files have been successfully indexed. If a file failed OCR silently in a previous run, delete its entry from the registry and restart:

```
health_ai/data/profiles/profile-1/processed_files.json
```

Remove the entry for the failing file and restart the server. Failed files are now not added to the registry automatically, so they will retry.

---

### Phone and laptop on same WiFi but still can't connect

1. Check your laptop firewall — port 8000 must be open for inbound connections
2. On macOS: System Settings → Network → Firewall → allow incoming connections for Python
3. On Windows: Windows Defender Firewall → allow Python through firewall
4. Confirm both devices are on the same network subnet (both should have `192.168.x.x` addresses)

---

## Model Details

| Property | Value |
|---|---|
| Model | Meta-Llama-3.1-8B-Instruct |
| Quantization | Q4_K_M (GGUF) |
| File size | ~4.9 GB |
| Context window | 8192 tokens |
| Runtime | llama-cpp-python |
| GPU offload | Enabled (`n_gpu_layers=-1`) — falls back to CPU if no GPU |
| Embedding model | all-MiniLM-L6-v2 (90 MB, 384 dimensions) |
| Embedding runtime | sentence-transformers |

---

## Privacy Notes

- **No internet required** after initial setup (model + packages downloaded once)
- **No cloud APIs** — LLM runs entirely on your laptop
- **Document text** is stored only in the server's local FAISS index and on the phone's AsyncStorage
- **Only retrieved chunks** (not full documents) are sent to the LLM for generation
- The server is only accessible within your local WiFi network — it is not exposed to the internet
