"""Ranking quality and group-level audit metrics."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _validate_inputs(
    records: pd.DataFrame,
    scores: Sequence[float],
    k: int,
) -> np.ndarray:
    score_values = np.asarray(scores, dtype=float)
    if len(records) == 0:
        raise ValueError("records must not be empty.")
    if score_values.ndim != 1 or len(score_values) != len(records):
        raise ValueError("scores must be a one-dimensional value per record.")
    if not np.isfinite(score_values).all():
        raise ValueError("scores must contain only finite values.")
    if not 1 <= k <= len(records):
        raise ValueError("k must be between 1 and the number of records.")
    if "qualified" not in records:
        raise ValueError("records must contain a 'qualified' column.")
    return score_values


def audit_by_group(
    records: pd.DataFrame,
    scores: Sequence[float],
    k: int,
    group_attribute: str,
    minimum_group_size: int = 30,
) -> pd.DataFrame:
    """Audit top-k selection and qualified-candidate recall by group."""
    score_values = _validate_inputs(records, scores, k)
    if group_attribute not in records:
        raise ValueError(f"Unknown group attribute: {group_attribute}")
    if minimum_group_size < 1:
        raise ValueError("minimum_group_size must be at least 1.")

    ranked_positions = np.argsort(-score_values, kind="stable")[:k]
    selected = np.zeros(len(records), dtype=bool)
    selected[ranked_positions] = True
    qualified = records["qualified"].to_numpy(dtype=bool)
    group_values = records[group_attribute].fillna("missing").to_numpy()

    rows = []
    for group in sorted(pd.unique(group_values), key=str):
        group_mask = group_values == group
        group_count = int(group_mask.sum())
        selected_count = int((group_mask & selected).sum())
        qualified_count = int((group_mask & qualified).sum())
        qualified_selected_count = int(
            (group_mask & selected & qualified).sum()
        )
        rows.append(
            {
                "group": str(group),
                "population": group_count,
                "small_group_warning": group_count < minimum_group_size,
                "selected": selected_count,
                "selection_rate_at_k": selected_count / group_count,
                "qualified": qualified_count,
                "qualified_selected": qualified_selected_count,
                "qualified_recall_at_k": (
                    qualified_selected_count / qualified_count
                    if qualified_count
                    else None
                ),
            }
        )

    highest_rate = max(row["selection_rate_at_k"] for row in rows)
    for row in rows:
        rate = float(row["selection_rate_at_k"])
        row["selection_ratio_vs_highest"] = (
            rate / highest_rate if highest_rate > 0 else None
        )
        row["meets_four_fifths_rule"] = (
            rate / highest_rate >= 0.8 if highest_rate > 0 else None
        )

    return pd.DataFrame(rows)


def ranking_metrics(
    records: pd.DataFrame,
    scores: Sequence[float],
    k: int,
) -> dict[str, float | int | None]:
    """Compute top-k precision/recall and NDCG using `qualified` as relevance."""
    score_values = _validate_inputs(records, scores, k)
    qualified = records["qualified"].to_numpy(dtype=bool)
    ranked_positions = np.argsort(-score_values, kind="stable")
    top_positions = ranked_positions[:k]
    hits = int(qualified[top_positions].sum())
    total_qualified = int(qualified.sum())

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    relevance = qualified[ranked_positions].astype(float)
    dcg = float(np.sum(relevance[:k] * discounts))
    ideal_relevance = np.sort(qualified.astype(float))[::-1][:k]
    ideal_dcg = float(np.sum(ideal_relevance * discounts))

    return {
        "k": k,
        "precision_at_k": hits / k,
        "recall_at_k": hits / total_qualified if total_qualified else None,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else None,
    }
