"""Train and compare historical-decision and job-signal ranking examples."""

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import make_synthetic_candidates
from .metrics import audit_by_group, ranking_metrics

MODEL_FEATURES = [
    "skills_score",
    "work_sample_score",
    "education_access",
    "experience_years",
]
JOB_SIGNAL_FEATURES = ["skills_score", "work_sample_score"]
AUDIT_ATTRIBUTES = ["gender_group", "age_group"]


@dataclass(frozen=True)
class ExperimentResult:
    """Test records, model scores, summary metrics and group audits."""

    ranked_test_records: pd.DataFrame
    summary: dict[str, dict[str, float | int | None]]
    group_audits: dict[str, pd.DataFrame]
    k: int


def run_experiment(
    n: int = 3000,
    random_state: int = 42,
    selection_fraction: float = 0.20,
) -> ExperimentResult:
    """Compare rankings learned from historical decisions and job signal.

    Neither protected attributes nor the generated labels represent real
    people, employers, or hiring outcomes.
    """
    if not 0.0 < selection_fraction <= 1.0:
        raise ValueError("selection_fraction must be greater than 0 and at most 1.")

    records = make_synthetic_candidates(n=n, random_state=random_state)
    train, test = train_test_split(
        records,
        test_size=0.30,
        random_state=random_state,
        stratify=records["qualified"],
    )
    k = max(1, int(round(len(test) * selection_fraction)))

    historical_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_state),
    )
    historical_model.fit(train[MODEL_FEATURES], train["historical_selection"])

    signal_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_state),
    )
    signal_model.fit(train[JOB_SIGNAL_FEATURES], train["qualified"])

    ranked = test[
        [
            "candidate_id",
            "gender_group",
            "age_group",
            "qualified",
            "historical_selection",
        ]
    ].copy()
    ranked["historical_decision_score"] = historical_model.predict_proba(
        test[MODEL_FEATURES]
    )[:, 1]
    ranked["job_signal_score"] = signal_model.predict_proba(
        test[JOB_SIGNAL_FEATURES]
    )[:, 1]

    summary = {}
    group_audits = {}
    for model_name, score_column in (
        ("historical_decision_model", "historical_decision_score"),
        ("job_signal_reference", "job_signal_score"),
    ):
        scores = ranked[score_column].to_numpy()
        summary[model_name] = ranking_metrics(ranked, scores, k)
        group_audits[model_name] = pd.concat(
            [
                audit_by_group(ranked, scores, k, attribute).assign(
                    attribute=attribute
                )
                for attribute in AUDIT_ATTRIBUTES
            ],
            ignore_index=True,
        )

    return ExperimentResult(
        ranked_test_records=ranked,
        summary=summary,
        group_audits=group_audits,
        k=k,
    )
