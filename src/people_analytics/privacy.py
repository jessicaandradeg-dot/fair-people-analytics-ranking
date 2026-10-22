"""Small pseudonymization and k-anonymity diagnostics for synthetic records."""

import hashlib
import hmac

import pandas as pd


def pseudonymize_candidate_ids(
    candidate_ids: pd.Series,
    secret_key: bytes,
) -> pd.Series:
    """Create keyed, stable pseudonyms without retaining source identifiers."""
    if len(secret_key) < 16:
        raise ValueError("secret_key must contain at least 16 bytes.")
    if candidate_ids.isna().any():
        raise ValueError("candidate_ids must not contain missing values.")
    return candidate_ids.astype(str).map(
        lambda value: "PHA-" + hmac.new(
            secret_key,
            value.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:24]
    )


def audit_k_anonymity(
    records: pd.DataFrame,
    quasi_identifiers: list[str],
    k: int = 5,
) -> tuple[dict[str, int | bool], pd.DataFrame]:
    """Count equivalence classes below k; this does not enforce suppression."""
    if records.empty:
        raise ValueError("records must not be empty.")
    if k < 2:
        raise ValueError("k must be at least 2.")
    if not quasi_identifiers:
        raise ValueError("At least one quasi-identifier is required.")
    missing = set(quasi_identifiers) - set(records.columns)
    if missing:
        raise ValueError(f"Records are missing quasi-identifiers: {sorted(missing)}")

    class_sizes = (
        records.assign(
            **{
                column: records[column].fillna("<missing>").astype(str)
                for column in quasi_identifiers
            }
        )
        .groupby(quasi_identifiers, dropna=False, observed=True)
        .size()
        .rename("class_size")
        .reset_index()
    )
    small_classes = class_sizes["class_size"] < k
    summary: dict[str, int | bool] = {
        "k": k,
        "equivalence_classes": int(len(class_sizes)),
        "minimum_class_size": int(class_sizes["class_size"].min()),
        "classes_below_k": int(small_classes.sum()),
        "records_in_classes_below_k": int(
            class_sizes.loc[small_classes, "class_size"].sum()
        ),
        "all_classes_meet_k": bool(not small_classes.any()),
    }
    return summary, class_sizes
