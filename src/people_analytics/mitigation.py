"""A narrow, illustrative demographic-parity post-processing exercise."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _allocate_group_selections(group_counts: np.ndarray, k: int) -> np.ndarray:
    """Allocate k slots approximately in proportion to group population."""
    exact = k * group_counts / group_counts.sum()
    allocation = np.floor(exact).astype(int)
    remainders = exact - allocation
    remaining = k - int(allocation.sum())
    for index in np.argsort(-remainders, kind="stable")[:remaining]:
        allocation[index] += 1
    return allocation


def demographic_parity_selection(
    records: pd.DataFrame,
    scores: Sequence[float],
    k: int,
    group_attribute: str,
) -> np.ndarray:
    """Select exactly k records with approximately equal group selection rates.

    Within each group, the highest scores are selected. Group-aware selection is
    deliberately included as a post-processing teaching example only; it is
    not a legal recommendation, validated hiring method, or production policy.
    """
    score_values = np.asarray(scores, dtype=float)
    if score_values.ndim != 1 or len(score_values) != len(records):
        raise ValueError("scores must be a one-dimensional value per record.")
    if not np.isfinite(score_values).all():
        raise ValueError("scores must contain only finite values.")
    if not 1 <= k <= len(records):
        raise ValueError("k must be between 1 and the number of records.")
    if group_attribute not in records:
        raise ValueError(f"Unknown group attribute: {group_attribute}")

    groups = records[group_attribute].fillna("missing").to_numpy()
    group_values, group_codes = np.unique(groups.astype(str), return_inverse=True)
    group_counts = np.bincount(group_codes, minlength=len(group_values))
    allocation = _allocate_group_selections(group_counts, k)
    selected = np.zeros(len(records), dtype=bool)

    for group_index, slots in enumerate(allocation):
        if slots == 0:
            continue
        positions = np.flatnonzero(group_codes == group_index)
        order = np.argsort(-score_values[positions], kind="stable")
        selected[positions[order[:slots]]] = True
    return selected


def selected_set_metrics(
    records: pd.DataFrame,
    selected: Sequence[bool],
) -> dict[str, float | int]:
    """Calculate quality and workload measures for a chosen top-k set."""
    selection = np.asarray(selected, dtype=bool)
    if selection.ndim != 1 or len(selection) != len(records):
        raise ValueError("selected must contain one boolean per record.")
    if not selection.any():
        raise ValueError("At least one record must be selected.")
    if "qualified" not in records or "review_minutes" not in records:
        raise ValueError("records must include qualified and review_minutes.")

    selected_records = records.loc[selection]
    selected_count = len(selected_records)
    qualified_count = int(selected_records["qualified"].astype(bool).sum())
    total_qualified = int(records["qualified"].astype(bool).sum())
    review_minutes = selected_records["review_minutes"].to_numpy(dtype=float)
    if not np.isfinite(review_minutes).all() or (review_minutes < 0).any():
        raise ValueError("review_minutes must be finite and non-negative.")
    return {
        "selected": selected_count,
        "qualified_selected": qualified_count,
        "precision_at_k": qualified_count / selected_count,
        "recall_at_k": qualified_count / total_qualified if total_qualified else 0.0,
        "total_review_minutes": float(review_minutes.sum()),
        "mean_review_minutes_per_selected": float(review_minutes.mean()),
    }
