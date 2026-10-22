import numpy as np
import pandas as pd
import pytest

from people_analytics.drift import compare_populations, population_stability_index


def test_psi_is_near_zero_for_identical_population():
    values = pd.Series(np.linspace(0, 1, 500))

    assert population_stability_index(values, values) == pytest.approx(0.0)


def test_psi_detects_synthetic_distribution_shift():
    reference = pd.Series(np.linspace(0, 1, 500))
    shifted = pd.Series(np.linspace(0.5, 1.5, 500))

    assert population_stability_index(reference, shifted) > 0.25


def test_categorical_psi_and_comparison_report_severity():
    reference = pd.DataFrame({"group": ["a"] * 90 + ["b"] * 10})
    current = pd.DataFrame({"group": ["a"] * 40 + ["b"] * 60})

    report = compare_populations(reference, current, ["group"])

    assert report.loc[0, "severity"] == "high"
    assert report.loc[0, "psi"] > 0.25


def test_psi_rejects_empty_populations():
    with pytest.raises(ValueError, match="must both contain records"):
        population_stability_index(pd.Series(dtype=float), pd.Series([1.0]))
