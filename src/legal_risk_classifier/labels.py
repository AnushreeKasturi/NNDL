"""The six risk-relevant clause categories this project classifies.

CUAD annotates 41 categories. We keep six, chosen for being commercially
material and reasonably well represented in the corpus. The mapping below is
the authoritative link between our display names and the category names CUAD
uses in its question ids.
"""

from __future__ import annotations

# Display name -> the category string CUAD uses after the "__" in a question id.
TARGET_LABELS: dict[str, str] = {
    "Cap on Liability": "Cap On Liability",
    "Non-Compete": "Non-Compete",
    "License Grant": "License Grant",
    "Audit Rights": "Audit Rights",
    "Termination for Convenience": "Termination For Convenience",
    "Insurance": "Insurance",
}

LABEL_NAMES: tuple[str, ...] = tuple(TARGET_LABELS)
LABEL_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}
NUM_LABELS = len(LABEL_NAMES)

# Reverse direction, for reading CUAD.
_CUAD_TO_LABEL: dict[str, str] = {v: k for k, v in TARGET_LABELS.items()}


def label_for_cuad_category(category: str) -> str | None:
    """Map a CUAD category name onto one of our six labels, or None."""
    return _CUAD_TO_LABEL.get(category.strip())
