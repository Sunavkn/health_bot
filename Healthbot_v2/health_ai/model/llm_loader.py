from llama_cpp import Llama
from threading import Lock
from health_ai.config.settings import LLM_MODEL_PATH


class LLMEngine:
    """
    Singleton LLM wrapper for Meta-Llama-3.1-8B-Instruct (GGUF).
    Uses the correct Llama-3 chat template:
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        self.llm = Llama(
            model_path=str(LLM_MODEL_PATH),
            n_ctx=8192,
            n_threads=8,
            n_batch=512,
            n_gpu_layers=-1,
            verbose=False,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 400,
    ) -> str:
        # Llama 3.1 Instruct chat template
        formatted_prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt.strip()}"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt.strip()}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        output = self.llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.2,
            stop=[
                "<|eot_id|>",
                "<|end_of_text|>",
                "<|start_header_id|>",
                "</s>",
            ],
        )

        raw = output["choices"][0]["text"].strip()

        # Hard-strip any leaked special tokens
        for tok in ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>",
                    "</s>", "<|end|>", "<|im_end|>"]:
            raw = raw.replace(tok, "").strip()

        return raw
