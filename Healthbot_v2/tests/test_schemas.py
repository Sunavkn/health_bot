from health_ai.ingestion.schemas import MedicalHistory

def test_medical_history_schema():
    data = {
        "patient_id": "123",
        "conditions": [
            {
                "name": "Hypertension",
                "diagnosed_date": "2020-01-01",
                "status": "active",
                "notes": "Under control"
            }
        ]
    }
    obj = MedicalHistory(**data)
    assert obj.patient_id == "123"
