"""
scoring.py
----------
Credit score utilities and helper functions.
The actual scores are pre-generated in preprocess.py using:

    credit_score = 700
                   - α * normalized_tmax
                   - β * normalized_ppt_variability
                   + random_noise

    α = 120, β = 80
    Scores clipped to [300, 850]
"""

import pandas as pd
import numpy as np


def score_color(score: float) -> str:
    """Return a hex color for a credit score value."""
    if score >= 700:
        return "#1a9641"
    elif score >= 650:
        return "#a6d96a"
    elif score >= 600:
        return "#fee08b"
    elif score >= 550:
        return "#f46d43"
    else:
        return "#d73027"


def score_label(score: float) -> str:
    """Human-readable credit risk label."""
    if score >= 700:
        return "Sangat Baik"
    elif score >= 650:
        return "Baik"
    elif score >= 600:
        return "Sedang"
    elif score >= 550:
        return "Berisiko"
    else:
        return "Tinggi"


def national_score(df: pd.DataFrame, year: int) -> float:
    """Weighted national credit score for a given year."""
    subset = df[df["year"] == year]
    return round(subset["credit_score"].mean(), 0)


def top_regions(df: pd.DataFrame, year: int, n: int = 10,
                level: str = "province") -> pd.DataFrame:
    """Top N regions by credit score."""
    subset = df[df["year"] == year].copy()
    if level == "province":
        agg = subset.groupby("province").agg(
            credit_score=("credit_score", "mean")
        ).reset_index()
        agg["credit_score"] = agg["credit_score"].round(0).astype(int)
        return agg.nlargest(n, "credit_score")
    else:
        return subset.nlargest(n, "credit_score")[
            ["province", "kabupaten", "credit_score"]
        ]
