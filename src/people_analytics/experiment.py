"""Train and compare historical-decision and job-signal ranking examples."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    AUDIT_ATTRIBUTES,
    JOB_GROUP_COLUMN,
    JOB_SIGNAL_FEATURES,
    MODEL_FEATURES,
    make_synthetic_candidates,
    prepare_and_integrate_synthetic_sources,
)
from .explainability import (
    coefficient_explanations,
    local_linear_explanations,
    shap_ranker_explanations,
)
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
    shap_explanations: pd.DataFrame
    k: int


def _grouped_ranking_metrics(
    records: pd.DataFrame,
    scores: pd.Series,
    selection_fraction: float,
) -> dict[str, float | int | None]:
    """Average ranking metrics computed independently within each vacancy."""
    per_vacancy = []
    for _, positions in records.groupby(JOB_GROUP_COLUMN, sort=True).groups.items():
        group = records.loc[positions].reset_index(drop=True)
        group_scores = scores.loc[positions].to_numpy(dtype=float)
        group_k = max(1, int(round(len(group) * selection_fraction)))
        per_vacancy.append(
            ranking_metrics(group, group_scores, min(group_k, len(group)))
        )
    recalls = [
        metric["recall_at_k"]
        for metric in per_vacancy
        if metric["recall_at_k"] is not None
    ]
    ndcgs = [
        metric["ndcg_at_k"]
        for metric in per_vacancy
        if metric["ndcg_at_k"] is not None
    ]
    return {
        "vacancies": len(per_vacancy),
        "mean_k_per_vacancy": float(np.mean([metric["k"] for metric in per_vacancy])),
        "precision_at_k": float(
            np.mean([metric["precision_at_k"] for metric in per_vacancy])
        ),
        "recall_at_k": float(np.mean(recalls)) if recalls else None,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else None,
    }


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
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_state,
    )
    train_positions, test_positions = next(
        splitter.split(
            records,
            records["qualified"],
            groups=records[JOB_GROUP_COLUMN],
        )
    )
    train = records.iloc[train_positions].copy()
    test = records.iloc[test_positions].copy()
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

    ranker_training = train.sort_values(JOB_GROUP_COLUMN, kind="stable")
    ranker_groups = (
        ranker_training.groupby(JOB_GROUP_COLUMN, sort=True)
        .size()
        .tolist()
    )
    ranker = LGBMRanker(
        objective="lambdarank",
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    )
    ranker.fit(
        ranker_training[JOB_SIGNAL_FEATURES],
        ranker_training["qualified"].astype(int),
        group=ranker_groups,
    )

    ranked = test.copy()
    ranked["historical_decision_score"] = historical_model.predict_proba(
        test[MODEL_FEATURES]
    )[:, 1]
    ranked["job_signal_score"] = signal_model.predict_proba(
        test[JOB_SIGNAL_FEATURES]
    )[:, 1]
    ranked["lightgbm_lambdarank_score"] = ranker.predict(
        test[JOB_SIGNAL_FEATURES]
    )

    summary = {}
    group_audits = {}
    for model_name, score_column in (
        ("historical_decision_model", "historical_decision_score"),
        ("job_signal_reference", "job_signal_score"),
    ):
        scores = ranked[score_column].to_numpy()
        if model_name == "historical_decision_model":
            summary[model_name] = ranking_metrics(ranked, scores, k)
        else:
            summary[model_name] = _grouped_ranking_metrics(
                ranked,
                ranked[score_column],
                selection_fraction,
            )
        group_audits[model_name] = pd.concat(
            [
                audit_by_group(ranked, scores, k, attribute).assign(
                    attribute=attribute
                )
                for attribute in AUDIT_ATTRIBUTES
            ],
            ignore_index=True,
        )

    summary["lightgbm_lambdarank"] = _grouped_ranking_metrics(
        ranked,
        ranked["lightgbm_lambdarank_score"],
        selection_fraction,
    )
    group_audits["lightgbm_lambdarank"] = pd.concat(
        [
            audit_by_group(
                ranked,
                ranked["lightgbm_lambdarank_score"].to_numpy(),
                k,
                attribute,
            ).assign(attribute=attribute)
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

    shap_explanations = shap_ranker_explanations(
        ranker,
        ranked.nlargest(10, "lightgbm_lambdarank_score"),
        JOB_SIGNAL_FEATURES,
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
        shap_explanations=shap_explanations,
        k=k,
    )
