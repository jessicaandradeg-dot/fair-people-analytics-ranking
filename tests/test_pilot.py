import pandas as pd
import pytest

from people_analytics.data import make_synthetic_candidates
from people_analytics.pilot import simulate_controlled_pilot


def test_synthetic_pilot_is_reproducible_and_reports_intervals():
    records = make_synthetic_candidates(n=500, random_state=9)
    scores = records["skills_score"].to_numpy()

    first_summary, first_assignments = simulate_controlled_pilot(
        records,
        scores,
        scores,
        random_state=9,
        bootstrap_iterations=100,
    )
    second_summary, second_assignments = simulate_controlled_pilot(
        records,
        scores,
        scores,
        random_state=9,
        bootstrap_iterations=100,
    )

    assert first_summary == second_summary
    pd.testing.assert_frame_equal(first_assignments, second_assignments)
    assert len(first_assignments) == 500
    assert set(first_assignments["arm"]) == {"control", "treatment"}
    assert len(first_summary["treatment_minus_control"]["precision_at_k_difference_ci_95"]) == 2


def test_pilot_rejects_too_few_bootstrap_iterations():
    records = make_synthetic_candidates(n=200)
    scores = records["skills_score"].to_numpy()

    with pytest.raises(ValueError, match="at least 100"):
        simulate_controlled_pilot(
            records,
            scores,
            scores,
            bootstrap_iterations=99,
        )
