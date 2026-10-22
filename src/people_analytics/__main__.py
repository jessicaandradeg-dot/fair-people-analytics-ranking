"""Run the synthetic ranking and fairness audit from the command line."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import AUDIT_ATTRIBUTES, JOB_SIGNAL_FEATURES, MODEL_FEATURES
from .drift import compare_populations
from .eda import exploratory_summary
from .experiment import run_experiment
from .pilot import simulate_controlled_pilot


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
    result.ranked_test_records.to_csv(
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
    result.local_explanations.to_json(
        args.output_dir / "local_model_explanations.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )
    regular_drift.to_csv(args.output_dir / "feature_drift_audit.csv", index=False)
    stress_drift.to_csv(
        args.output_dir / "synthetic_drift_stress_test.csv",
        index=False,
    )
    pilot_assignments.to_csv(
        args.output_dir / "synthetic_pilot_assignments.csv",
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
        "mitigation_group_audit": result.mitigation_audit.to_dict(orient="records"),
        "explainability": {
            "method": (
                "Standardized logistic-regression coefficients and exact linear "
                "contributions to log-odds; not SHAP and not causal explanations."
            ),
            "global_coefficients": result.coefficient_explanations.to_dict(
                orient="records"
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
