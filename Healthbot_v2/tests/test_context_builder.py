from health_ai.rag.context_builder import ContextBuilder


def test_context_budget():
    builder = ContextBuilder()
    retrieved = [{"text": "medical history " * 300, "metadata": {}}]
    history = ["previous message " * 100]
    query = "What is happening?"

    context = builder.build_context(retrieved, history, query)

    assert len(context) > 0
    assert "PATIENT DATA" in context
    assert "QUESTION" in context


def test_context_sections_present():
    builder = ContextBuilder()
    retrieved = [{"text": "Hemoglobin: 14.5 g/dL", "metadata": {}}]
    history = ["User: hello", "Assistant: hi"]
    query = "What is my hemoglobin?"

    context = builder.build_context(retrieved, history, query)

    assert "PATIENT DATA" in context
    assert "CONVERSATION HISTORY" in context
    assert "QUESTION" in context
    assert "Hemoglobin" in context
