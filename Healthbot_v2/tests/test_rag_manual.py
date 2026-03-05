from health_ai.api.rag_pipeline import RAGPipeline

def test_rag_pipeline_smoke():
    pipeline = RAGPipeline()
    response = pipeline.run("I have high blood pressure.")
    assert isinstance(response, str)
