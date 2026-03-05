RED_FLAGS = [
    "chest pain",
    "shortness of breath",
    "severe headache",
    "confusion",
    "blood in stool",
    "suicidal"
]


def detect_red_flags(text: str):
    lower = text.lower()
    return any(flag in lower for flag in RED_FLAGS)
