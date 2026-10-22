"""Small, transparent explanations for standardized logistic models."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def coefficient_explanations(
    model: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return global standardized coefficient magnitudes and directions."""
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    coefficients = classifier.coef_[0]
    if len(coefficients) != len(feature_names):
        raise ValueError("feature_names must match the fitted model features.")

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "standardized_coefficient": coefficients,
                "absolute_coefficient": np.abs(coefficients),
                "direction": np.where(coefficients >= 0, "increases_score", "decreases_score"),
                "scale_mean": scaler.mean_,
                "scale_std": scaler.scale_,
            }
        )
        .sort_values("absolute_coefficient", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def local_linear_explanations(
    model: Pipeline,
    records: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 3,
) -> pd.DataFrame:
    """Explain predictions as standardized linear contributions to log-odds."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    missing = set(feature_names) - set(records.columns)
    if missing:
        raise ValueError(f"Records are missing features: {sorted(missing)}")
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    if classifier.coef_.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the fitted model features.")

    scaled = scaler.transform(records[feature_names])
    contributions = scaled * classifier.coef_[0]
    intercept = float(classifier.intercept_[0])
    output = []
    for row_index, row_contributions in enumerate(contributions):
        top_positions = np.argsort(-np.abs(row_contributions), kind="stable")[:top_n]
        output.append(
            {
                "record_index": row_index,
                "base_log_odds": intercept,
                "top_contributions": [
                    {
                        "feature": feature_names[position],
                        "contribution_to_log_odds": float(row_contributions[position]),
                    }
                    for position in top_positions
                ],
            }
        )
    return pd.DataFrame(output)
