"""Agora's ``--judge exact``: normalized exact match, 1.0 or 0.0.

The normalizer is copied verbatim from vendor/agora-code/evaluation/src/judge.py
(``_normalize_for_exact``, ``ExactJudge.judge``); do not "improve" it or the
score stops being comparable to published Agora numbers. It is strict on
purpose: units, sign, thousands separators and rounding must all match.
"""

from __future__ import annotations

import re


def _normalize_for_exact(s: str) -> str:
    s = (s or "").strip()
    s = s.strip("`'\"")
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def score(gold: str, prediction: str) -> float:
    if not (prediction or "").strip():
        return 0.0
    return float(_normalize_for_exact(prediction) == _normalize_for_exact(gold))
