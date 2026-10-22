"""An offline A/B pilot simulation for ranking quality and review effort."""

import numpy as np
import pandas as pd

from .mitigation import demographic_parity_selection, selected_set_metrics


def _bootstrap_difference(
    treatment: np.ndarray,
    control: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float, float]:
    if treatment.size == 0 or control.size == 0:
        raise ValueError("Both pilot arms must contain selected records.")
    differences = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        treatment_sample = rng.choice(treatment, size=treatment.size, replace=True)
        control_sample = rng.choice(control, size=control.size, replace=True)
        differences[iteration] = treatment_sample.mean() - control_sample.mean()
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return float(treatment.mean() - control.mean()), float(lower), float(upper)


def simulate_controlled_pilot(
    records: pd.DataFrame,
    baseline_scores: np.ndarray,
    treatment_scores: np.ndarray,
    selection_fraction: float = 0.20,
    group_attribute: str = "gender_group",
    random_state: int = 42,
    bootstrap_iterations: int = 1000,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Randomize synthetic records to a baseline or mitigation pilot arm.

    The simulated pilot is an offline design exercise, not a live experiment.
    """
    if not 0.0 < selection_fraction <= 1.0:
        raise ValueError("selection_fraction must be greater than 0 and at most 1.")
    if bootstrap_iterations < 100:
        raise ValueError("bootstrap_iterations must be at least 100.")
    if group_attribute not in records:
        raise ValueError(f"Unknown group attribute: {group_attribute}")
    if not {"qualified", "review_minutes"}.issubset(records.columns):
        raise ValueError("records must include qualified and review_minutes.")

    baseline = np.asarray(baseline_scores, dtype=float)
    treatment = np.asarray(treatment_scores, dtype=float)
    if (
        baseline.ndim != 1
        or treatment.ndim != 1
        or len(baseline) != len(records)
        or len(treatment) != len(records)
    ):
        raise ValueError("Each pilot arm needs one score per input record.")
    if not np.isfinite(baseline).all() or not np.isfinite(treatment).all():
        raise ValueError("Pilot scores must be finite.")

    rng = np.random.default_rng(random_state)
    randomized_positions = rng.permutation(len(records))
    split = len(records) // 2
    control_positions = np.sort(randomized_positions[:split])
    treatment_positions = np.sort(randomized_positions[split:])
    selections = np.zeros(len(records), dtype=bool)
    arm_names = np.full(len(records), "control", dtype=object)
    arm_names[treatment_positions] = "treatment"

    control_k = max(1, int(round(len(control_positions) * selection_fraction)))
    control_order = np.argsort(
        -baseline[control_positions],
        kind="stable",
    )
    selections[control_positions[control_order[:control_k]]] = True

    treatment_k = max(1, int(round(len(treatment_positions) * selection_fraction)))
    treatment_records = records.iloc[treatment_positions].reset_index(drop=True)
    selections[treatment_positions] = demographic_parity_selection(
        treatment_records,
        treatment[treatment_positions],
        treatment_k,
        group_attribute,
    )

    control_selected = records.iloc[control_positions[selections[control_positions]]]
    treatment_selected = records.iloc[
        treatment_positions[selections[treatment_positions]]
    ]
    control_quality = control_selected["qualified"].to_numpy(dtype=float)
    treatment_quality = treatment_selected["qualified"].to_numpy(dtype=float)
    control_effort = control_selected["review_minutes"].to_numpy(dtype=float)
    treatment_effort = treatment_selected["review_minutes"].to_numpy(dtype=float)

    quality_difference, quality_low, quality_high = _bootstrap_difference(
        treatment_quality,
        control_quality,
        rng,
        bootstrap_iterations,
    )
    effort_difference, effort_low, effort_high = _bootstrap_difference(
        treatment_effort,
        control_effort,
        rng,
        bootstrap_iterations,
    )
    assignments = records[["candidate_id", group_attribute, "qualified", "review_minutes"]].copy()
    assignments["arm"] = arm_names
    assignments["selected_for_review"] = selections

    summary = {
        "notice": (
            "Offline simulation on synthetic records. This is not an online pilot, "
            "real hiring experiment, or evidence of real-world impact."
        ),
        "seed": random_state,
        "randomized_records": len(records),
        "selection_fraction_per_arm": selection_fraction,
        "bootstrap_iterations": bootstrap_iterations,
        "control_policy": "unmitigated_baseline_top_k",
        "treatment_policy": "synthetic_gender_group_demographic_parity_postprocessing",
        "control": selected_set_metrics(
            records.iloc[control_positions].reset_index(drop=True),
            selections[control_positions],
        ),
        "treatment": selected_set_metrics(
            records.iloc[treatment_positions].reset_index(drop=True),
            selections[treatment_positions],
        ),
        "treatment_minus_control": {
            "precision_at_k_difference": quality_difference,
            "precision_at_k_difference_ci_95": [quality_low, quality_high],
            "mean_review_minutes_difference": effort_difference,
            "mean_review_minutes_difference_ci_95": [effort_low, effort_high],
        },
    }
    return summary, assignments
