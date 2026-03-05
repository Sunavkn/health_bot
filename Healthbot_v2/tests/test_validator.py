import pytest
from health_ai.ingestion.validator import validate_medical_history
from health_ai.core.exceptions import SchemaValidationError

def test_validator_success():
    data = {
        "patient_id": "1",
        "conditions": []
    }
    result = validate_medical_history(data)
    assert result.patient_id == "1"

def test_validator_failure():
    data = {"wrong": "format"}
    with pytest.raises(SchemaValidationError):
        validate_medical_history(data)
