import numpy as np
from health_ai.embeddings.embedder import EmbeddingModel

def test_embedding_shape():
    model = EmbeddingModel()
    vec = model.embed("This is a medical test sentence.")
    assert isinstance(vec, np.ndarray)
    assert vec.shape[1] == 384

def test_embedding_similarity():
    model = EmbeddingModel()
    v1 = model.embed("Patient has high blood pressure.")
    v2 = model.embed("The patient suffers from hypertension.")
    similarity = np.dot(v1[0], v2[0])
    assert similarity > 0.5
