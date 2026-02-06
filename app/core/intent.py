HIGH_RISK_KEYWORDS = [
    "diagnose", "diagnosis", "do i have",
    "prescribe", "medication", "medicine",
    "dosage", "treatment plan", "should i take"
]

def classify_intent(text: str) -> str:
    text = text.lower()
    for kw in HIGH_RISK_KEYWORDS:
        if kw in text:
            return "disallowed"
    return "informational"

