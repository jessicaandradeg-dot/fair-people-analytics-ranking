import pandas as pd
import pytest

from people_analytics.data import make_synthetic_candidates


def test_synthetic_data_is_reproducible_and_only_contains_fake_ids():
    first = make_synthetic_candidates(n=150, random_state=7)
    second = make_synthetic_candidates(n=150, random_state=7)

    pd.testing.assert_frame_equal(first, second)
    assert first["candidate_id"].str.startswith("SYN-").all()
    assert len(first) == 150


def test_synthetic_data_rejects_too_small_samples():
    with pytest.raises(ValueError, match="at least 100"):
        make_synthetic_candidates(n=99)
