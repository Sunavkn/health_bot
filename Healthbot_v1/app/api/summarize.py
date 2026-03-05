from fastapi import APIRouter, HTTPException
from app.core.pdf_reader import read_pdf_text
from app.core.chunking import chunk_text
from app.core.summarize import summarize_text
from app.core.inference import local_infer
from app.core.prompts import SUMMARIZE_SYSTEM_PROMPT
from app.schemas.summarize import SummarizeResponse
from app.core.paths import SAMPLE_PDF_PATH
from app.core.summary_cleaner import clean_summary_output

router = APIRouter()

PDF_PATH = SAMPLE_PDF_PATH


@router.post("/", response_model=SummarizeResponse)
def summarize():
    try:
        text = read_pdf_text(PDF_PATH)
    except Exception:
        raise HTTPException(status_code=500, detail="Unable to read PDF file.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="PDF contains no readable text.")

    chunks = chunk_text(text)

    # Step 1: summarize each chunk
    chunk_summaries = []
    for chunk in chunks[:3]:  # cap for now
        chunk_summaries.append(summarize_text(chunk))

    combined_text = "\n\n".join(chunk_summaries)

    # Step 2: FINAL consolidation pass (THIS WAS MISSING)
    final_summary = local_infer(
    f"{SUMMARIZE_SYSTEM_PROMPT}\n\n"
    "IMPORTANT:\n"
    "- Do NOT include thoughts, plans, reasoning, or analysis.\n"
    "- Do NOT include headings like 'Plan', 'Refinement', or 'Formatting'.\n"
    "- Respond ONLY with the final patient summary.\n\n"
    "Create one clean, patient-friendly summary from the text below. "
    "Remove repetition and group related tests.\n\n"
    f"{combined_text}",
    max_tokens=700
)
    final_summary = clean_summary_output(final_summary)


    return SummarizeResponse(
        summary=final_summary.strip(),
        disclaimer="This summary is for informational purposes only."
    )
