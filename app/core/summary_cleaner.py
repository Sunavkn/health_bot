import re

def clean_summary_output(text: str) -> str:
    if not text:
        return ""

    banned_starts = (
        "thought",
        "plan:",
        "refinement:",
        "formatting:",
        "patient information:",
        "**plan**",
        "**refinement**",
    )

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()

        if not stripped:
            continue

        if any(lower.startswith(b) for b in banned_starts):
            continue

        # remove accidental markdown artifacts
        stripped = re.sub(r"\*\*(plan|refinement|formatting).*?\*\*", "", stripped, flags=re.I)

        lines.append(stripped)

    return "\n".join(lines).strip()
