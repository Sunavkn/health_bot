from health_ai.ingestion.ingest import IngestionEngine
from health_ai.api.rag_pipeline import RAGPipeline
import uuid


def test_profile_isolation():

    profile1 = str(uuid.uuid4())
    profile2 = str(uuid.uuid4())

    engine1 = IngestionEngine(profile1)
    engine2 = IngestionEngine(profile2)

    engine1.ingest_medical_history({
        "patient_id": "1",
        "conditions": [
            {
                "name": "Hypertension",
                "diagnosed_date": "2020-01-01",
                "status": "active",
                "notes": "Profile 1"
            }
        ]
    })

    engine2.ingest_medical_history({
        "patient_id": "2",
        "conditions": [
            {
                "name": "Diabetes",
                "diagnosed_date": "2021-01-01",
                "status": "active",
                "notes": "Profile 2"
            }
        ]
    })

    pipeline1 = RAGPipeline(profile1)
    response1 = pipeline1.run("blood sugar")

    assert "diabetes" not in response1.lower()
