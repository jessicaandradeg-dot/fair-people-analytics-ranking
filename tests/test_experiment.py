import pandas as pd
import pytest

from people_analytics.experiment import run_experiment


def test_experiment_is_reproducible_and_audits_groups():
    first = run_experiment(n=500, random_state=17)
    second = run_experiment(n=500, random_state=17)

    assert first.k == second.k
    assert first.summary == second.summary
    pd.testing.assert_frame_equal(
        first.group_audits["historical_decision_model"],
        second.group_audits["historical_decision_model"],
    )
    assert set(first.group_audits) == {
        "historical_decision_model",
        "job_signal_reference",
        "lightgbm_lambdarank",
    }
    assert first.summary["lightgbm_lambdarank"]["vacancies"] > 0
    assert set(first.shap_explanations["feature"]) == {
        "skills_score",
        "work_sample_score",
    }
    assert first.shap_explanations["shap_value"].notna().all()
    assert set(first.training_records["job_id"]).isdisjoint(
        set(first.ranked_test_records["job_id"])
    )
    assert set(first.group_audits["historical_decision_model"]["attribute"]) == {
        "gender_group",
        "race_group_synthetic",
        "age_group",
    }


def test_experiment_rejects_invalid_selection_fraction():
    with pytest.raises(ValueError, match="selection_fraction"):
        run_experiment(n=200, selection_fraction=0)
