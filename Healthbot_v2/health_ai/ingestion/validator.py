from pydantic import ValidationError
from health_ai.ingestion.schemas import *
from health_ai.core.exceptions import SchemaValidationError

def validate_medical_history(data):
    try:
        return MedicalHistory(**data)
    except ValidationError as e:
        raise SchemaValidationError(str(e))

def validate_hospitalizations(data):
    try:
        return Hospitalizations(**data)
    except ValidationError as e:
        raise SchemaValidationError(str(e))

def validate_prescriptions(data):
    try:
        return Prescriptions(**data)
    except ValidationError as e:
        raise SchemaValidationError(str(e))

def validate_lab_reports(data):
    try:
        return LabReports(**data)
    except ValidationError as e:
        raise SchemaValidationError(str(e))

def validate_family_history(data):
    try:
        return FamilyHistory(**data)
    except ValidationError as e:
        raise SchemaValidationError(str(e))

def validate_daily_log(data):
    try:
        return DailyLog(**data)
    except ValidationError as e:
        raise SchemaValidationError(str(e))
