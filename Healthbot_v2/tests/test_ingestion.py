from health_ai.ingestion.ingest import IngestionEngine

def test_ingest_medical_history():

    engine = IngestionEngine()

    data = {
        "patient_id": "123",
        "conditions": [
            {
                "name": "Hypertension",
                "diagnosed_date": "2022-01-01",
                "status": "active",
                "notes": "Under monitoring"
            }
        ]
    }

    engine.ingest_medical_history(data)
    assert True
