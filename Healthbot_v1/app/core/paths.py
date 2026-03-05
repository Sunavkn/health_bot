from pathlib import Path

# aichatbot2/
BASE_DIR = Path(__file__).resolve().parents[2]

# Directories
APP_DIR = BASE_DIR / "app"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "sample_reports"

# Files
MEDGEMMA_MODEL_PATH = MODELS_DIR / "medgemma-1.5-4b-it-Q4_K_M.gguf"
SAMPLE_PDF_PATH = DATA_DIR / "sterling-accuris-pathology-sample-report-unlocked.pdf"
