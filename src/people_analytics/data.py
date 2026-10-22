"""Generate a reproducible dataset for fairness and ranking demonstrations."""

import numpy as np
import pandas as pd


def make_synthetic_candidates(
    n: int = 3000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return synthetic candidate records with intentionally imperfect labels.

    Protected attributes are included only for auditing. They are not model
    features. Historical decisions deliberately include group and proxy effects
    so the example can illustrate why omitting protected columns is not enough
    to guarantee a fair outcome.
    """
    if n < 100:
        raise ValueError("n must be at least 100 to support a meaningful demo.")

    rng = np.random.default_rng(random_state)
    gender_group = rng.choice(
        ["women", "men", "nonbinary"],
        size=n,
        p=[0.47, 0.47, 0.06],
    )
    age = rng.integers(22, 61, size=n)
    age_group = np.where(age < 40, "under_40", "40_and_over")

    skills_score = np.clip(rng.normal(0.62, 0.16, size=n), 0.0, 1.0)
    work_sample_score = np.clip(
        0.65 * skills_score + rng.normal(0.23, 0.16, size=n),
        0.0,
        1.0,
    )

    women = gender_group == "women"
    nonbinary = gender_group == "nonbinary"
    older = age_group == "40_and_over"
    education_access = np.clip(
        rng.normal(
            0.72 - 0.12 * women - 0.08 * nonbinary - 0.06 * older,
            0.16,
            size=n,
        ),
        0.0,
        1.0,
    )
    experience_years = np.clip(
        (age - 22) * 0.42
        + rng.normal(0.0, 3.0, size=n)
        - 1.3 * women
        - 0.8 * nonbinary,
        0.0,
        30.0,
    )

    job_signal = 0.65 * skills_score + 0.35 * work_sample_score
    qualification_probability = 1.0 / (
        1.0 + np.exp(-9.0 * (job_signal - 0.60))
    )
    qualified = rng.binomial(1, qualification_probability)

    historical_log_odds = (
        -1.2
        + 2.3 * qualified
        + 0.9 * education_access
        + 0.025 * experience_years
        - 0.30 * women
        - 0.25 * nonbinary
        - 0.20 * older
    )
    historical_probability = 1.0 / (1.0 + np.exp(-historical_log_odds))
    historical_selection = rng.binomial(1, historical_probability)

    return pd.DataFrame(
        {
            "candidate_id": [f"SYN-{index:05d}" for index in range(n)],
            "skills_score": skills_score,
            "work_sample_score": work_sample_score,
            "education_access": education_access,
            "experience_years": experience_years,
            "gender_group": gender_group,
            "age_group": age_group,
            "qualified": qualified,
            "historical_selection": historical_selection,
        }
    )
