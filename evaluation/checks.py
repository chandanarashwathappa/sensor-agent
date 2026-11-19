"""Grading of agent answers (deterministic, no LLM-as-judge)."""
from __future__ import annotations

import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    """All numbers in a text; thousands separators like 10,000 are handled."""
    out = []
    for m in _NUM.findall(text):
        s = m.replace(",", "")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def check_number(answer: str, expected: float, tolerance: float) -> bool:
    """True if any number in the answer is within tolerance of the expected value.
    Rounding is allowed: 3.39 passes for 3.39, and 46.1 passes for 46.08 with tolerance 0.1."""
    tol = max(tolerance, 1e-9)
    return any(abs(v - expected) <= tol + 1e-9 for v in numbers_in(answer))


def check_keywords(answer: str, groups: list[list[str]]) -> bool:
    """Every group needs at least one match. Single capital letters must be whole words."""
    for group in groups:
        found = False
        for word in group:
            if len(word) == 1 and word.isupper():
                if re.search(rf"\b{word}\b", answer):
                    found = True
            elif word.lower() in answer.lower():
                found = True
        if not found:
            return False
    return True


def grade(question: dict, answer: str) -> bool:
    if question["kind"] == "number":
        return check_number(answer, float(question["expected"]), float(question["tolerance"]))
    return check_keywords(answer, question["expected"])
