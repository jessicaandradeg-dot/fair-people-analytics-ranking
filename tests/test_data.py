import pandas as pd
import pytest

from people_analytics.data import make_synthetic_candidates
from people_analytics.data import prepare_and_integrate_synthetic_sources


def test_synthetic_data_is_reproducible_and_only_contains_fake_ids():
    first = make_synthetic_candidates(n=150, random_state=7)
    second = make_synthetic_candidates(n=150, random_state=7)

    pd.testing.assert_frame_equal(first, second)
    assert first["candidate_id"].str.startswith("SYN-").all()
    assert len(first) == 150
    assert first["review_minutes"].between(3, 15).all()
    assert set(first["race_group_synthetic"]).issubset(
        {"synthetic_group_a", "synthetic_group_b", "synthetic_group_c"}
    )


def test_synthetic_data_rejects_too_small_samples():
    with pytest.raises(ValueError, match="at least 100"):
        make_synthetic_candidates(n=99)


def test_source_integration_joins_profiles_and_assessments_by_unique_id():
    records = make_synthetic_candidates(n=150, random_state=7)

    integrated = prepare_and_integrate_synthetic_sources(records)

    assert len(integrated) == len(records)
    assert integrated["candidate_id"].is_unique
    assert integrated["work_sample_score"].equals(records["work_sample_score"])


def test_source_integration_rejects_duplicate_candidate_ids():
    records = make_synthetic_candidates(n=150, random_state=7)
    records.loc[1, "candidate_id"] = records.loc[0, "candidate_id"]

    with pytest.raises(ValueError, match="candidate_id must be present and unique"):
        prepare_and_integrate_synthetic_sources(records)
