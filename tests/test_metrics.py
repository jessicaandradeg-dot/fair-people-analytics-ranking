import numpy as np
import pandas as pd
import pytest

from people_analytics.metrics import audit_by_group, ranking_metrics


def test_ranking_metrics_for_perfect_order():
    records = pd.DataFrame(
        {
            "qualified": [1, 1, 0, 0],
            "group": ["a", "b", "a", "b"],
        }
    )

    metrics = ranking_metrics(records, [0.9, 0.8, 0.2, 0.1], k=2)

    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["ndcg_at_k"] == 1.0


def test_audit_reports_selection_and_qualified_recall_by_group():
    records = pd.DataFrame(
        {
            "qualified": [1, 1, 0, 0],
            "group": ["a", "b", "a", "b"],
        }
    )

    audit = audit_by_group(records, [0.9, 0.8, 0.2, 0.1], k=2, group_attribute="group")
    group_a = audit.set_index("group").loc["a"]
    group_b = audit.set_index("group").loc["b"]

    assert group_a["selection_rate_at_k"] == 0.5
    assert group_b["selection_rate_at_k"] == 0.5
    assert group_a["qualified_recall_at_k"] == 1.0
    assert group_b["qualified_recall_at_k"] == 1.0
    assert group_a["small_group_warning"]
    assert group_b["small_group_warning"]
    assert group_a["selection_ratio_vs_highest"] == 1.0
    assert group_a["meets_four_fifths_rule"]
    assert group_b["meets_four_fifths_rule"]


def test_audit_rejects_invalid_minimum_group_size():
    records = pd.DataFrame({"qualified": [1, 0], "group": ["a", "b"]})
    with pytest.raises(ValueError, match="minimum_group_size"):
        audit_by_group(records, [0.8, 0.2], k=1, group_attribute="group", minimum_group_size=0)


@pytest.mark.parametrize("k", [0, 5])
def test_metrics_reject_invalid_k(k):
    records = pd.DataFrame({"qualified": [1, 0]})
    with pytest.raises(ValueError, match="k must be"):
        ranking_metrics(records, np.array([0.8, 0.2]), k)


def test_metrics_reject_non_finite_scores():
    records = pd.DataFrame({"qualified": [1, 0]})
    with pytest.raises(ValueError, match="finite"):
        ranking_metrics(records, [float("nan"), 0.2], k=1)
