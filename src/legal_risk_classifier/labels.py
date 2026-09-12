from __future__ import annotations

TARGET_LABELS = [
    "Cap on Liability",
    "Non-Compete",
    "License Grant",
    "Audit Rights",
    "Termination for Convenience",
    "Insurance",
]

_NORMALIZED_LOOKUP = {
    "cap on liability": "Cap on Liability",
    "cap-on-liability": "Cap on Liability",
    "liability cap": "Cap on Liability",
    "non-compete": "Non-Compete",
    "non compete": "Non-Compete",
    "no-solicit of employees": "Non-Compete",
    "license grant": "License Grant",
    "license-grant": "License Grant",
    "audit rights": "Audit Rights",
    "audit-rights": "Audit Rights",
    "termination for convenience": "Termination for Convenience",
    "termination-for-convenience": "Termination for Convenience",
    "insurance": "Insurance",
}

LABEL_TO_INDEX = {label: i for i, label in enumerate(TARGET_LABELS)}


def normalize_label(raw: str) -> str | None:
    key = " ".join(raw.strip().lower().replace("_", " ").replace("/", " ").split())
    if key in _NORMALIZED_LOOKUP:
        return _NORMALIZED_LOOKUP[key]
    for candidate in TARGET_LABELS:
        if key == candidate.lower():
            return candidate
    return None

