from app.core.inference import local_infer
from app.core.prompts import SUMMARIZE_SYSTEM_PROMPT

def summarize_text(text: str) -> str:
    prompt = (
        f"{SUMMARIZE_SYSTEM_PROMPT}\n\n"
        "Summarize the following medical document content clearly:\n\n"
        f"{text}"
    )

    summary = local_infer(prompt, max_tokens=600)
    return summary
