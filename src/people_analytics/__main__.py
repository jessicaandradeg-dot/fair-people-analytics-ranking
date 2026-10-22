"""Run the synthetic ranking and fairness audit from the command line."""

import argparse
import json
import os
from pathlib import Path
import secrets

import numpy as np
import pandas as pd

from .data import AUDIT_ATTRIBUTES, JOB_SIGNAL_FEATURES, MODEL_FEATURES
from .drift import compare_populations
from .eda import exploratory_summary
from .experiment import run_experiment
from .pilot import simulate_controlled_pilot
from .privacy import audit_k_anonymity, pseudonymize_candidate_ids


def _pseudonymized_copy(
    frame: pd.DataFrame,
    secret_key: bytes,
) -> pd.DataFrame:
    """Return a report frame with candidate IDs replaced by keyed pseudonyms."""
    output = frame.copy()
    if "candidate_id" in output:
        output["candidate_id"] = pseudonymize_candidate_ids(
            output["candidate_id"],
            secret_key,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a synthetic People Analytics ranking audit."
    )
    parser.add_argument("--n", type=int, default=3000, help="Number of synthetic records.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--selection-fraction",
        type=float,
        default=0.20,
        help="Fraction of the test pool included in the top-k audit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated CSV and JSON reports.",
    )
    args = parser.parse_args()

    result = run_experiment(
        n=args.n,
        random_state=args.seed,
        selection_fraction=args.selection_fraction,
    )
    configured_key = os.environ.get("PEOPLE_ANALYTICS_PSEUDONYM_KEY")
    pseudonym_key = (
        configured_key.encode("utf-8")
        if configured_key is not None
        else secrets.token_bytes(32)
    )
    pseudonym_key_source = (
        "environment variable"
        if configured_key is not None
        else "ephemeral random key (IDs are not linkable across runs)"
    )
    k_anonymity_summary, k_anonymity_classes = audit_k_anonymity(
        result.ranked_test_records,
        ["gender_group", "race_group_synthetic", "age_group"],
        k=5,
    )
    stress_population = result.ranked_test_records.copy()
    stress_population["education_access"] = np.clip(
        stress_population["education_access"] + 0.20,
        0.0,
        1.0,
    )
    stress_population["experience_years"] = np.clip(
        stress_population["experience_years"] + 4.0,
        0.0,
        30.0,
    )
    drift_columns = list(dict.fromkeys(MODEL_FEATURES + AUDIT_ATTRIBUTES))
    regular_drift = compare_populations(
        result.training_records,
        result.ranked_test_records,
        drift_columns,
    )
    stress_drift = compare_populations(
        result.ranked_test_records,
        stress_population,
        drift_columns,
    ).assign(scenario="synthetic_shift_stress_test")
    pilot_summary, pilot_assignments = simulate_controlled_pilot(
        result.ranked_test_records.reset_index(drop=True),
        result.ranked_test_records["historical_decision_score"].to_numpy(),
        result.ranked_test_records["historical_decision_score"].to_numpy(),
        selection_fraction=args.selection_fraction,
        random_state=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _pseudonymized_copy(
        result.ranked_test_records,
        pseudonym_key,
    ).to_csv(
        args.output_dir / "synthetic_test_scores.csv",
        index=False,
    )
    for model_name, audit in result.group_audits.items():
        audit.to_csv(args.output_dir / f"{model_name}_fairness_audit.csv", index=False)
    result.mitigation_audit.to_csv(
        args.output_dir / "demographic_parity_postprocessing_audit.csv",
        index=False,
    )
    result.coefficient_explanations.to_csv(
        args.output_dir / "model_coefficient_explanations.csv",
        index=False,
    )
    _pseudonymized_copy(
        result.local_explanations,
        pseudonym_key,
    ).to_json(
        args.output_dir / "local_model_explanations.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )
    _pseudonymized_copy(
        result.shap_explanations,
        pseudonym_key,
    ).to_csv(
        args.output_dir / "lambdarank_shap_explanations.csv",
        index=False,
    )
    regular_drift.to_csv(args.output_dir / "feature_drift_audit.csv", index=False)
    stress_drift.to_csv(
        args.output_dir / "synthetic_drift_stress_test.csv",
        index=False,
    )
    _pseudonymized_copy(pilot_assignments, pseudonym_key).to_csv(
        args.output_dir / "synthetic_pilot_assignments.csv",
        index=False,
    )
    k_anonymity_classes.to_csv(
        args.output_dir / "synthetic_k_anonymity_classes.csv",
        index=False,
    )

    eda_report = exploratory_summary(result.training_records)
    (args.output_dir / "exploratory_summary.json").write_text(
        json.dumps(eda_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "synthetic_pilot_summary.json").write_text(
        json.dumps(pilot_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    output = {
        "notice": (
            "Synthetic teaching demonstration only. Not for real hiring decisions, "
            "LGPD compliance claims, or production monitoring."
        ),
        "seed": args.seed,
        "test_records": len(result.ranked_test_records),
        "training_vacancies": int(result.training_records["job_id"].nunique()),
        "test_vacancies": int(result.ranked_test_records["job_id"].nunique()),
        "k": result.k,
        "ranking_metrics": result.summary,
        "demographic_parity_postprocessing": {
            "method": (
                "Group-aware top-k allocation proportional to group size; "
                "illustrative technique, not a legal or production recommendation."
            ),
            "selection_metrics": result.mitigation_metrics,
        },
        "feature_drift": {
            "method": "Population Stability Index (PSI); feature-distribution triage only.",
            "train_vs_test": regular_drift.to_dict(orient="records"),
            "synthetic_shift_stress_test": stress_drift.to_dict(orient="records"),
        },
        "controlled_pilot_simulation": pilot_summary,
        "group_audits": {
            name: audit.to_dict(orient="records")
            for name, audit in result.group_audits.items()
        },
        "four_fifths_rule": {
            "interpretation": (
                "Selection-rate ratio versus the highest-rate group; descriptive "
                "screening heuristic only, not a legal test or fairness guarantee."
            ),
            "audits": {
                name: audit[
                    [
                        "attribute",
                        "group",
                        "selection_rate_at_k",
                        "selection_ratio_vs_highest",
                        "meets_four_fifths_rule",
                        "population",
                    ]
                ].to_dict(orient="records")
                for name, audit in result.group_audits.items()
            },
        },
        "privacy_demonstrations": {
            "pseudonymization": {
                "method": "HMAC-SHA256 with a key supplied via environment or generated for this run.",
                "key_source": pseudonym_key_source,
                "notice": "Pseudonymization is not anonymization; never publish the key.",
            },
            "k_anonymity": {
                "quasi_identifiers": [
                    "gender_group",
                    "race_group_synthetic",
                    "age_group",
                ],
                **k_anonymity_summary,
                "notice": (
                    "Synthetic equivalence-class diagnostic only; no suppression "
                    "or privacy guarantee is implemented."
                ),
            },
        },
        "mitigation_group_audit": result.mitigation_audit.to_dict(orient="records"),
        "explainability": {
            "method": (
                "Standardized logistic coefficients/linear log-odds contributions "
                "and SHAP contributions for the LightGBM LambdaRank score. Neither "
                "is causal or a justification for hiring decisions."
            ),
            "global_coefficients": result.coefficient_explanations.to_dict(
                orient="records"
            ),
            "lambdarank_mean_absolute_shap": (
                result.shap_explanations.groupby("feature")[
                    "absolute_shap_value"
                ]
                .mean()
                .sort_values(ascending=False)
                .to_dict()
            ),
        },
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nReports saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
