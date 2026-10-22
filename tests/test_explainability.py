import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from people_analytics.data import MODEL_FEATURES, make_synthetic_candidates
from people_analytics.explainability import (
    coefficient_explanations,
    local_linear_explanations,
)
from people_analytics.experiment import run_experiment


def test_global_coefficients_and_local_contributions_match_model_scores():
    records = make_synthetic_candidates(n=500, random_state=5)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=5),
    ).fit(records[MODEL_FEATURES], records["qualified"])
    coefficients = coefficient_explanations(model, MODEL_FEATURES)
    local = local_linear_explanations(
        model,
        records.head(5),
        MODEL_FEATURES,
        top_n=len(MODEL_FEATURES),
    )
    contributions = np.array(
        [sum(item["contribution_to_log_odds"] for item in row) for row in local["top_contributions"]]
    )
    reconstructed = local["base_log_odds"].to_numpy() + contributions
    actual = model.decision_function(records[MODEL_FEATURES].head(5))

    assert len(coefficients) == len(MODEL_FEATURES)
    assert len(local) == 5
    np.testing.assert_allclose(reconstructed, actual)


def test_experiment_returns_explanations_for_both_models():
    experiment = run_experiment(n=500, random_state=5)

    assert set(experiment.coefficient_explanations["model"]) == {
        "historical_decision_model",
        "job_signal_reference",
    }
    assert set(experiment.local_explanations["model"]) == {
        "historical_decision_model",
        "job_signal_reference",
    }
    assert len(experiment.local_explanations) == 20
    contributions_by_model = experiment.local_explanations.groupby("model")[
        "top_contributions"
    ].apply(lambda values: {len(item) for item in values})
    assert contributions_by_model["historical_decision_model"] == {3}
    assert contributions_by_model["job_signal_reference"] == {2}
