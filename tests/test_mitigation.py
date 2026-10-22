import numpy as np
import pandas as pd
import pytest

from people_analytics.mitigation import (
    demographic_parity_selection,
    selected_set_metrics,
)


def test_group_selection_postprocessing_approximately_equalizes_rates():
    records = pd.DataFrame(
        {
            "group": ["a"] * 80 + ["b"] * 20,
            "qualified": [1] * 40 + [0] * 40 + [1] * 10 + [0] * 10,
            "review_minutes": [8.0] * 100,
        }
    )
    scores = np.concatenate((np.linspace(1.0, 0.2, 80), np.linspace(0.8, 0.1, 20)))

    selected = demographic_parity_selection(records, scores, 20, "group")
    report = records.assign(selected=selected).groupby("group")["selected"].mean()

    assert int(selected.sum()) == 20
    assert report.loc["a"] == pytest.approx(0.2)
    assert report.loc["b"] == pytest.approx(0.2)


def test_selected_set_metrics_report_quality_and_effort():
    records = pd.DataFrame(
        {
            "qualified": [1, 1, 0, 0],
            "review_minutes": [5.0, 7.0, 9.0, 11.0],
        }
    )

    metrics = selected_set_metrics(records, [True, True, False, False])

    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mean_review_minutes_per_selected"] == 6.0
    assert metrics["total_review_minutes"] == 12.0
