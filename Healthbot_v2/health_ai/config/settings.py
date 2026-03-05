from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
STATIC_DATA_DIR = DATA_DIR / "static"
DYNAMIC_DATA_DIR = DATA_DIR / "dynamic"
LOG_DIR = BASE_DIR / "logs"

# ── Token budgets ─────────────────────────────────────────────────────────────
# Summary queries now feed all pages directly (up to 9000 chars ≈ ~2200 tokens).
# Standard RAG queries use top_k=8 chunks at ~400 words each ≈ ~700 tokens.
MAX_CONTEXT_TOKENS = 6000
SYSTEM_PROMPT_TOKENS = 400
RETRIEVAL_TOKENS = 2800
HISTORY_TOKENS = 200
QUERY_TOKENS = 200

LLM_MODEL_PATH = BASE_DIR / "model" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

for path in [DATA_DIR, STATIC_DATA_DIR, DYNAMIC_DATA_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)
