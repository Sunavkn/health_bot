from typing import List
from health_ai.config.settings import (
    MAX_CONTEXT_TOKENS,
    RETRIEVAL_TOKENS,
    HISTORY_TOKENS,
    QUERY_TOKENS,
)


def approximate_token_count(text: str) -> int:
    return int(len(text.split()) * 1.3)


class ContextBuilder:

    def __init__(self):
        self.max_tokens = MAX_CONTEXT_TOKENS

    def build_context(
        self,
        retrieved_chunks: List[dict],
        conversation_history: List[str],
        user_query: str,
    ) -> str:
        """
        Builds the user-prompt context string fed to the LLM.
        Structure:
            [PATIENT DATA]
            <retrieved chunk 1>
            ---
            <retrieved chunk 2>
            ...
            [CONVERSATION HISTORY]
            <last N turns>
            [QUESTION]
            <user_query>
        """
        parts = []

        # ── Retrieved patient data ───────────────────────────────────
        retrieval_budget = RETRIEVAL_TOKENS
        data_parts = []

        for chunk in retrieved_chunks:
            chunk_text = chunk["text"]
            tokens = approximate_token_count(chunk_text)
            if tokens <= retrieval_budget:
                data_parts.append(chunk_text)
                retrieval_budget -= tokens
            else:
                break

        if data_parts:
            parts.append("[PATIENT DATA]\n" + "\n---\n".join(data_parts))

        # ── Conversation history (most recent first, then reversed) ──
        history_budget = HISTORY_TOKENS
        history_parts = []

        for msg in reversed(conversation_history):
            tokens = approximate_token_count(msg)
            if tokens <= history_budget:
                history_parts.append(msg)
                history_budget -= tokens
            else:
                break

        if history_parts:
            parts.append("[CONVERSATION HISTORY]\n" + "\n".join(reversed(history_parts)))

        # ── User question ────────────────────────────────────────────
        if approximate_token_count(user_query) > QUERY_TOKENS:
            user_query = " ".join(user_query.split()[:QUERY_TOKENS])

        parts.append(f"[QUESTION]\n{user_query}")

        final_context = "\n\n".join(parts)

        # Safety trim
        if approximate_token_count(final_context) > MAX_CONTEXT_TOKENS:
            final_context = final_context[: MAX_CONTEXT_TOKENS * 4]

        return final_context
