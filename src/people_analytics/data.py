"""Generate a reproducible dataset for fairness and ranking demonstrations."""

import numpy as np
import pandas as pd

MODEL_FEATURES = [
    "skills_score",
    "work_sample_score",
    "education_access",
    "experience_years",
]
JOB_SIGNAL_FEATURES = ["skills_score", "work_sample_score"]
AUDIT_ATTRIBUTES = ["gender_group", "race_group_synthetic", "age_group"]
JOB_GROUP_COLUMN = "job_id"


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
    race_group_synthetic = rng.choice(
        ["synthetic_group_a", "synthetic_group_b", "synthetic_group_c"],
        size=n,
        p=[0.60, 0.25, 0.15],
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
    group_b = race_group_synthetic == "synthetic_group_b"
    group_c = race_group_synthetic == "synthetic_group_c"
    older = age_group == "40_and_over"
    job_number = rng.integers(0, 12, size=n)
    job_ids = np.array([f"VAC-{number:02d}" for number in job_number])
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

    job_skill_weight = np.linspace(0.35, 0.75, 12)[job_number]
    job_signal = (
        job_skill_weight * skills_score
        + (1.0 - job_skill_weight) * work_sample_score
    )
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
        - 0.20 * group_b
        - 0.10 * group_c
        - 0.20 * older
    )
    historical_probability = 1.0 / (1.0 + np.exp(-historical_log_odds))
    historical_selection = rng.binomial(1, historical_probability)
    review_minutes = np.clip(
        rng.normal(7.0 + 3.0 * (1.0 - work_sample_score), 1.5, size=n),
        3.0,
        15.0,
    )

    return pd.DataFrame(
        {
            "candidate_id": [f"SYN-{index:05d}" for index in range(n)],
            JOB_GROUP_COLUMN: job_ids,
            "skills_score": skills_score,
            "work_sample_score": work_sample_score,
            "education_access": education_access,
            "experience_years": experience_years,
            "review_minutes": review_minutes,
            "gender_group": gender_group,
            "race_group_synthetic": race_group_synthetic,
            "age_group": age_group,
            "qualified": qualified,
            "historical_selection": historical_selection,
        }
    )


def prepare_and_integrate_synthetic_sources(
    records: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and join two synthetic tables using a one-to-one candidate key."""
    required = {
        "candidate_id",
        JOB_GROUP_COLUMN,
        "skills_score",
        "education_access",
        "experience_years",
        "review_minutes",
        "gender_group",
        "race_group_synthetic",
        "age_group",
        "work_sample_score",
        "qualified",
        "historical_selection",
    }
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Synthetic source data is missing columns: {sorted(missing)}")
    if records.empty:
        raise ValueError("Synthetic source data must not be empty.")
    if records["candidate_id"].isna().any() or records["candidate_id"].duplicated().any():
        raise ValueError("candidate_id must be present and unique in each source.")

    profile_columns = [
        "candidate_id",
        JOB_GROUP_COLUMN,
        "skills_score",
        "education_access",
        "experience_years",
        "review_minutes",
        "gender_group",
        "race_group_synthetic",
        "age_group",
    ]
    assessment_columns = [
        "candidate_id",
        "work_sample_score",
        "qualified",
        "historical_selection",
    ]
    profile_source = records[profile_columns].copy()
    assessment_source = records[assessment_columns].copy()
    integrated = profile_source.merge(
        assessment_source,
        on="candidate_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = integrated["_merge"] != "both"
    if unmatched.any():
        raise ValueError(
            f"Sources contain {int(unmatched.sum())} unmatched candidate records."
        )
    return integrated.drop(columns="_merge")
