def format_medical_history(obj):
    texts = []
    for c in obj.conditions:
        texts.append(
            f"Condition: {c.name}. "
            f"Diagnosed: {c.diagnosed_date}. "
            f"Status: {c.status}. "
            f"Notes: {c.notes}."
        )
    return "\n".join(texts)


def format_hospitalizations(obj):
    texts = []
    for h in obj.hospitalizations:
        texts.append(
            f"Hospital: {h.hospital}. "
            f"Reason: {h.reason}. "
            f"Admission: {h.admission_date}. "
            f"Discharge: {h.discharge_date}. "
            f"Summary: {h.summary}."
        )
    return "\n".join(texts)


def format_prescriptions(obj):
    texts = []
    for p in obj.prescriptions:
        texts.append(
            f"Medication: {p.drug_name}. "
            f"Dosage: {p.dosage}. "
            f"Frequency: {p.frequency}. "
            f"Prescribed for: {p.prescribed_for}. "
            f"Start: {p.start_date}. "
            f"End: {p.end_date}."
        )
    return "\n".join(texts)


def format_lab_reports(obj):
    texts = []
    for r in obj.lab_reports:
        values = ", ".join([f"{k}: {v}" for k, v in r.values.items()])
        texts.append(
            f"Lab Test: {r.test_name}. "
            f"Date: {r.date}. "
            f"Values: {values}. "
            f"Normal Range: {r.normal_range}. "
            f"Notes: {r.notes}."
        )
    return "\n".join(texts)


def format_family_history(obj):
    texts = []
    for f in obj.family_history:
        texts.append(
            f"Family relation: {f.relation}. "
            f"Condition: {f.condition}. "
            f"Diagnosed at age: {f.age_at_diagnosis}."
        )
    return "\n".join(texts)


def format_daily_log(obj):
    symptoms = ", ".join(
        [f"{s.name} ({s.severity}) for {s.duration_hours} hours"
         for s in obj.symptoms]
    ) if obj.symptoms else "None"

    return (
        f"Date: {obj.date}. "
        f"Steps: {obj.activity.steps}. "
        f"Calories burned: {obj.activity.calories_burned}. "
        f"Water intake: {obj.water_intake_liters} liters. "
        f"Symptoms: {symptoms}."
    )
