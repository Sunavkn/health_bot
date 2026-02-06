from llama_cpp import Llama
import os

from app.core.paths import MEDGEMMA_MODEL_PATH

MODEL_PATH = str(MEDGEMMA_MODEL_PATH)


llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
    temperature=0.2,
    top_p=0.9,
)

def local_infer(
    prompt: str,
    max_tokens: int = 512,
    system_prompt: str | None = None
) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical information assistant. "
                    "Provide general, educational medical information only. "
                    "Do not diagnose, prescribe medication, or suggest treatments. "
                    "Respond only with the final patient-facing answer. "
                    "Use simple language and short paragraphs."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
    )

    return response["choices"][0]["message"]["content"] or ""
