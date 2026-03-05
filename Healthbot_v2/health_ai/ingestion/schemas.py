from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import date

class Condition(BaseModel):
    name: str
    diagnosed_date: date
    status: str
    notes: Optional[str] = ""

class MedicalHistory(BaseModel):
    patient_id: str
    conditions: List[Condition]

class Hospitalization(BaseModel):
    hospital: str
    reason: str
    admission_date: date
    discharge_date: date
    summary: str

class Hospitalizations(BaseModel):
    hospitalizations: List[Hospitalization]

class Prescription(BaseModel):
    drug_name: str
    dosage: str
    frequency: str
    start_date: date
    end_date: Optional[date]
    prescribed_for: str

class Prescriptions(BaseModel):
    prescriptions: List[Prescription]

class LabReport(BaseModel):
    test_name: str
    date: date
    values: Dict[str, str]
    normal_range: str
    notes: Optional[str] = ""

class LabReports(BaseModel):
    lab_reports: List[LabReport]

class FamilyHistoryEntry(BaseModel):
    relation: str
    condition: str
    age_at_diagnosis: int

class FamilyHistory(BaseModel):
    family_history: List[FamilyHistoryEntry]

class Symptom(BaseModel):
    name: str
    severity: str
    duration_hours: int

class Activity(BaseModel):
    steps: int
    calories_burned: int

class DailyLog(BaseModel):
    date: date
    activity: Activity
    water_intake_liters: float
    food_intake: Optional[List[Dict[str, int]]] = []
    symptoms: Optional[List[Symptom]] = []
