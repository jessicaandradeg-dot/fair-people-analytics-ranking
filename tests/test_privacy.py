import pandas as pd
import pytest

from people_analytics.privacy import (
    audit_k_anonymity,
    pseudonymize_candidate_ids,
)


def test_pseudonyms_are_stable_within_key_and_change_with_another_key():
    ids = pd.Series(["SYN-1", "SYN-2"])

    first = pseudonymize_candidate_ids(ids, b"a" * 32)
    repeated = pseudonymize_candidate_ids(ids, b"a" * 32)
    changed = pseudonymize_candidate_ids(ids, b"b" * 32)

    assert first.equals(repeated)
    assert not first.equals(changed)
    assert first.str.startswith("PHA-").all()


def test_pseudonymization_rejects_short_or_missing_key():
    with pytest.raises(ValueError, match="at least 16 bytes"):
        pseudonymize_candidate_ids(pd.Series(["SYN-1"]), b"short")
    with pytest.raises(ValueError, match="missing values"):
        pseudonymize_candidate_ids(pd.Series(["SYN-1", None]), b"a" * 32)


def test_k_anonymity_diagnostic_counts_small_equivalence_classes():
    records = pd.DataFrame({"age_band": ["a", "a", "a", "b"], "region": ["x"] * 4})

    summary, classes = audit_k_anonymity(records, ["age_band", "region"], k=3)

    assert summary["equivalence_classes"] == 2
    assert summary["classes_below_k"] == 1
    assert summary["records_in_classes_below_k"] == 1
    assert not summary["all_classes_meet_k"]
    assert set(classes["class_size"]) == {1, 3}
