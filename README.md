🩺 HealthBot v1 – On-Device Medical Assistant API

HealthBot v1 is a privacy-first, safety-focused medical assistant backend designed to run entirely on-device using a local medical LLM (MedGemma).
It provides general medical information and patient-friendly medical document summaries while strictly avoiding diagnosis, prescriptions, or treatment plans.

This project is built as part of the Health Digital Twin initiative at PES University (CAVE Labs).

🚀 Features
✅ Conversational Medical Assistant (/chat)

Explains medical terms and conditions in simple language

Lists common symptoms when appropriate

Provides general lifestyle and self-care guidance (rest, hydration, warm fluids)

Never diagnoses

Never recommends medicines

Gracefully handles diagnostic questions (e.g., “Do I have fever?”)

✅ Medical Document Summarization (/summarize)

Reads medical PDFs (lab reports, pathology reports)

Extracts key values and normal reference ranges

Groups related tests (CBC, Blood Sugar, Lipids, Thyroid, etc.)

Removes lab methodology, interferences, and references

Produces a clean, patient-friendly summary

Informational only (non-diagnostic)

🔐 Privacy & Safety

100% on-device inference

No cloud calls

No user data stored

Strict intent guardrails

Prompt-based safety control (no fine-tuning on user data)

🧠 Model & Runtime

Model: MedGemma 1.5 4B (GGUF)

Inference: llama-cpp-python

Execution: CPU-only, local runtime

Safety: Prompt-controlled behavior (no RLHF, no training)

📁 Project Structure
healthbot_v1/
├── app/
│   ├── api/
│   │   ├── chat.py              # Chat endpoint
│   │   └── summarize.py         # PDF summarization endpoint
│   ├── core/
│   │   ├── inference.py         # Local LLM inference
│   │   ├── prompts.py           # System prompts
│   │   ├── intent.py            # Intent classification
│   │   ├── guardrails.py        # Safety enforcement
│   │   ├── pdf_reader.py        # PDF text extraction
│   │   ├── chunking.py          # Text chunking
│   │   ├── summarize.py         # Chunk-level summarization
│   │   ├── summary_cleaner.py   # Final summary cleanup
│   │   └── paths.py             # Relative paths
│   ├── schemas/
│   │   ├── chat.py
│   │   └── summarize.py
│   ├── tests/
│   └── main.py                  # FastAPI app entry
├── models/
│   └── medgemma-1.5-4b-it-Q4_K_M.gguf
├── pdf/
│   └── sample_report.pdf
├── venv/
├── README.md

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/CAVE-PESU/Health-Digital-Twin.git
cd Health-Digital-Twin
git checkout Sunav

2️⃣ Create & Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate   # macOS / Linux

3️⃣ Install Dependencies
pip install --upgrade pip
pip install fastapi uvicorn llama-cpp-python pypdf


⚠️ Make sure your system supports llama-cpp-python (Apple Silicon or x86_64 CPU).

4️⃣ Place the Model

Download the MedGemma GGUF model and place it here:

models/medgemma-1.5-4b-it-Q4_K_M.gguf


Or set an environment variable:

export MEDGEMMA_MODEL_PATH=models/medgemma-1.5-4b-it-Q4_K_M.gguf

▶️ Running the Server
uvicorn app.main:app --reload


Server will start at:

http://127.0.0.1:8000


Health check:

curl http://127.0.0.1:8000/health

🧪 API Usage
🔹 1. Chat Endpoint

Endpoint

POST /chat


Request

curl -X POST http://127.0.0.1:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is fever?"
  }'


Example Behaviors

✅ “What is asthma?”

✅ “What are common symptoms of diabetes?”

⚠️ “Do I have fever?” → Safe fallback, no diagnosis

❌ “What medicine should I take?” → Blocked

🔹 2. PDF Summarization Endpoint

Endpoint

POST /summarize


(Currently uses a configured sample PDF path)

Request

curl -X POST http://127.0.0.1:8000/summarize/


Response

Grouped test results

Patient values + normal ranges

Clean, readable summary

Informational disclaimer

🛡️ Safety Rules (Strictly Enforced)

HealthBot v1 will not:

Diagnose conditions

Prescribe medications

Suggest treatment plans

Interpret results clinically

Store user data

If a user asks a diagnostic question:

The assistant explains it cannot diagnose, then provides general information and common symptoms.

📦 Versioning

healthbot_v1

Conversational medical info

PDF summarization

On-device inference

Prompt-based safety

No UI, backend-only

Future versions may add:

Domain-based document retrieval

Multi-document timelines

Structured medical summaries

UI integration

🧑‍💻 Author

Sunav
B.Tech Computer Science
PES University – CAVE Labs

⚠️ Disclaimer

This project is for educational and research purposes only.
It does not replace professional medical advice, diagnosis, or treatment.