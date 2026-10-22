"""Feature-distribution drift checks using population stability index (PSI)."""

import numpy as np
import pandas as pd


def _severity(psi: float) -> str:
    if psi < 0.10:
        return "low"
    if psi < 0.25:
        return "investigate"
    return "high"


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    smoothing: float = 0.5,
) -> float:
    """Compute PSI using reference quantiles for numeric data or categories."""
    if reference.empty or current.empty:
        raise ValueError("reference and current must both contain records.")
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    if smoothing <= 0:
        raise ValueError("smoothing must be greater than 0.")

    if pd.api.types.is_numeric_dtype(reference) and pd.api.types.is_numeric_dtype(current):
        ref_values = reference.to_numpy(dtype=float)
        cur_values = current.to_numpy(dtype=float)
        if not np.isfinite(ref_values).all() or not np.isfinite(cur_values).all():
            raise ValueError("PSI inputs must contain finite numeric values.")
        quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
        internal_edges = np.unique(np.quantile(ref_values, quantiles))
        edges = np.concatenate(([-np.inf], internal_edges, [np.inf]))
        ref_counts = np.histogram(ref_values, bins=edges)[0].astype(float)
        cur_counts = np.histogram(cur_values, bins=edges)[0].astype(float)
    else:
        ref_values = reference.fillna("missing").astype(str)
        cur_values = current.fillna("missing").astype(str)
        categories = sorted(set(ref_values) | set(cur_values))
        ref_counts = np.array([(ref_values == c).sum() for c in categories], dtype=float)
        cur_counts = np.array([(cur_values == c).sum() for c in categories], dtype=float)

    ref_share = (ref_counts + smoothing) / (ref_counts.sum() + smoothing * len(ref_counts))
    cur_share = (cur_counts + smoothing) / (cur_counts.sum() + smoothing * len(cur_counts))
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def compare_populations(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Calculate one PSI and a triage label per requested feature."""
    if reference.empty or current.empty:
        raise ValueError("reference and current populations must both be non-empty.")
    missing = set(columns) - set(reference.columns) | (set(columns) - set(current.columns))
    if missing:
        raise ValueError(f"Populations are missing columns: {sorted(missing)}")

    rows = []
    for column in columns:
        psi = population_stability_index(reference[column], current[column])
        rows.append(
            {
                "feature": column,
                "psi": psi,
                "severity": _severity(psi),
                "reference_count": len(reference),
                "current_count": len(current),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
