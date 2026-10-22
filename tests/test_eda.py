import pytest

from people_analytics.data import make_synthetic_candidates
from people_analytics.eda import exploratory_summary


def test_eda_reports_missingness_groups_and_label_rates():
    records = make_synthetic_candidates(n=300, random_state=11)

    summary = exploratory_summary(records)

    assert summary["records"] == 300
    assert summary["missing_values_by_feature"]["skills_score"] == 0
    assert summary["qualified_rate"] == records["qualified"].mean()
    assert set(summary["group_label_rates"]) == {
        "gender_group",
        "race_group_synthetic",
        "age_group",
    }


def test_eda_rejects_missing_audit_fields():
    records = make_synthetic_candidates(n=200).drop(columns="age_group")

    with pytest.raises(ValueError, match="missing required fields"):
        exploratory_summary(records)
