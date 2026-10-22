"""Train and compare historical-decision and job-signal ranking examples."""

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    AUDIT_ATTRIBUTES,
    JOB_SIGNAL_FEATURES,
    MODEL_FEATURES,
    make_synthetic_candidates,
    prepare_and_integrate_synthetic_sources,
)
from .explainability import coefficient_explanations, local_linear_explanations
from .metrics import audit_by_group, ranking_metrics
from .mitigation import demographic_parity_selection, selected_set_metrics


@dataclass(frozen=True)
class ExperimentResult:
    """Training/test records, model scores, and concise audit outputs."""

    training_records: pd.DataFrame
    ranked_test_records: pd.DataFrame
    summary: dict[str, dict[str, float | int | None]]
    group_audits: dict[str, pd.DataFrame]
    mitigation_selection: pd.Series
    mitigation_metrics: dict[str, float | int]
    mitigation_audit: pd.DataFrame
    coefficient_explanations: pd.DataFrame
    local_explanations: pd.DataFrame
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

    records = prepare_and_integrate_synthetic_sources(
        make_synthetic_candidates(n=n, random_state=random_state)
    )
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

    ranked = test.copy()
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

    mitigation = demographic_parity_selection(
        ranked,
        ranked["historical_decision_score"].to_numpy(),
        k,
        "gender_group",
    )
    mitigation_selection = pd.Series(
        mitigation,
        index=ranked.index,
        name="demographic_parity_selected",
    )
    mitigation_metrics = selected_set_metrics(ranked, mitigation)
    mitigation_audit = pd.concat(
        [
            audit_by_group(
                ranked,
                mitigation.astype(float),
                k,
                attribute,
            ).assign(attribute=attribute)
            for attribute in AUDIT_ATTRIBUTES
        ],
        ignore_index=True,
    )

    coefficient_explanation_frames = []
    local_explanation_frames = []
    for model_name, model, model_features, score_column in (
        (
            "historical_decision_model",
            historical_model,
            MODEL_FEATURES,
            "historical_decision_score",
        ),
        (
            "job_signal_reference",
            signal_model,
            JOB_SIGNAL_FEATURES,
            "job_signal_score",
        ),
    ):
        coefficient_explanation_frames.append(
            coefficient_explanations(model, model_features).assign(model=model_name)
        )
        selected_for_explanation = ranked.nlargest(10, score_column)
        local_explanation_frames.append(
            local_linear_explanations(
                model,
                selected_for_explanation,
                model_features,
            )
            .assign(
                model=model_name,
                candidate_id=selected_for_explanation["candidate_id"].to_numpy(),
            )
        )

    return ExperimentResult(
        training_records=train.reset_index(drop=True),
        ranked_test_records=ranked,
        summary=summary,
        group_audits=group_audits,
        mitigation_selection=mitigation_selection,
        mitigation_metrics=mitigation_metrics,
        mitigation_audit=mitigation_audit,
        coefficient_explanations=pd.concat(
            coefficient_explanation_frames,
            ignore_index=True,
        ),
        local_explanations=pd.concat(local_explanation_frames, ignore_index=True),
        k=k,
    )
