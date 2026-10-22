"""Run the synthetic ranking and fairness audit from the command line."""

import argparse
import json
from pathlib import Path

from .experiment import run_experiment


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.ranked_test_records.to_csv(
        args.output_dir / "synthetic_test_scores.csv",
        index=False,
    )
    for model_name, audit in result.group_audits.items():
        audit.to_csv(args.output_dir / f"{model_name}_fairness_audit.csv", index=False)

    output = {
        "notice": "Synthetic teaching demonstration; not for real hiring decisions.",
        "seed": args.seed,
        "test_records": len(result.ranked_test_records),
        "k": result.k,
        "ranking_metrics": result.summary,
        "group_audits": {
            name: audit.to_dict(orient="records")
            for name, audit in result.group_audits.items()
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
