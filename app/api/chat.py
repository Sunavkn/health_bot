from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.core.intent import classify_intent
from app.core.inference import local_infer
import re

router = APIRouter()

OTC_KEYWORDS = [
    "acetaminophen",
    "paracetamol",
    "ibuprofen",
    "advil",
    "tylenol",
    "aspirin",
    "naproxen",
    "decongestant",
    "antihistamine",
    "cough syrup",
]


def clean_output(text: str) -> str:
    if not text:
        return ""

    lines = []
    for raw in text.splitlines():
        line = raw.strip()

        # Drop junk bullets / noise
        if not line or line in {"•", "-", "*"}:
            continue

        # Normalize bullets
        if line.startswith(("•", "-", "*")):
            line = "- " + line.lstrip("•-* ").strip()

        # Remove OTC meds
        if any(k in line.lower() for k in OTC_KEYWORDS):
            continue

        lines.append(line)

    result = []
    prev_list = False

    for line in lines:
        is_list = line.startswith("- ")

        if is_list and not prev_list:
            result.append("")

        if not is_list and prev_list:
            result.append("")

        result.append(line)
        prev_list = is_list

    return "\n".join(result).strip()


def diagnostic_prompt(question: str) -> str:
    q = question.lower().replace("do i have", "what is")
    return f"{q.strip()} and common symptoms"


@router.post("/")
def chat(req: ChatRequest):
    intent = classify_intent(req.question)

    if intent == "disallowed":
        raw = local_infer(diagnostic_prompt(req.question))
        cleaned = clean_output(raw)

        final_text = (
            "I can’t help with diagnosing a condition.\n"
            "However, here is general information and common symptoms that may be associated.\n\n"
            f"{cleaned}"
        )
    else:
        raw = local_infer(req.question)
        final_text = clean_output(raw)

    return ChatResponse(
        reply=final_text,
        disclaimer="This information is for general education only.",
        confidence="medium",
    )
