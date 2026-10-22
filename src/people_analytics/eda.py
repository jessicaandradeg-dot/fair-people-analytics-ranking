"""Compact exploratory summaries for the synthetic candidate dataset."""

import pandas as pd

from .data import AUDIT_ATTRIBUTES, JOB_SIGNAL_FEATURES, MODEL_FEATURES


def exploratory_summary(records: pd.DataFrame) -> dict[str, object]:
    """Summarize distributions, missingness, label rates, and correlations."""
    features = list(dict.fromkeys(MODEL_FEATURES + JOB_SIGNAL_FEATURES))
    missing = set(features + ["qualified", *AUDIT_ATTRIBUTES]) - set(records.columns)
    if missing:
        raise ValueError(f"Records are missing required fields: {sorted(missing)}")

    numeric = records[features].describe().round(4)
    group_rates: dict[str, list[dict[str, object]]] = {}
    for attribute in AUDIT_ATTRIBUTES:
        rows = []
        values = records[attribute].fillna("missing")
        for group in sorted(values.unique(), key=str):
            group_records = records.loc[values == group]
            rows.append(
                {
                    "group": str(group),
                    "count": int(len(group_records)),
                    "qualified_rate": float(group_records["qualified"].mean()),
                    "historical_selection_rate": float(
                        group_records["historical_selection"].mean()
                    ),
                }
            )
        group_rates[attribute] = rows

    return {
        "records": len(records),
        "feature_columns": features,
        "missing_values_by_feature": {
            name: int(count) for name, count in records[features].isna().sum().items()
        },
        "numeric_summary": numeric.to_dict(),
        "feature_correlation": records[features].corr().round(4).to_dict(),
        "qualified_rate": float(records["qualified"].mean()),
        "historical_selection_rate": float(records["historical_selection"].mean()),
        "group_label_rates": group_rates,
    }
