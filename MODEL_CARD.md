# Model card: synthetic vacancy ranking prototype

## Intended use

Educational demonstration of grouped ranking, fairness measurement, explanation, drift triage, and pilot design using generated data. It is not a candidate assessment or hiring system.

## Models

- Historical-decision logistic regression: learns the generated historical-selection label from four numeric features.
- Job-signal logistic regression: reference baseline on skills and work-sample scores.
- LightGBM LambdaRank: ranking model trained with binary qualification labels and vacancy query groups.
- A separate top-k group-aware post-processing example is an illustration, not a recommended hiring policy.

## Data and labels

All records, labels, group attributes, vacancy IDs, and outcomes are synthetic. Historical selection intentionally includes group and proxy effects. Qualification is sampled from a simulated vacancy-specific skills/work-sample signal. Neither label is ground truth about real people or job performance.

## Evaluation

Vacancy IDs are held out between training and test. The logistic reference and LambdaRank are evaluated with macro-averaged per-vacancy precision@k, recall@k, and NDCG@k. Selection-rate audits, four-fifths ratios, and SHAP summaries are descriptive outputs over the synthetic test set.

## Limitations and risks

- No real candidate data, job requirements, hiring outcomes, legal review, or organizational context is included.
- Group categories are artificial and must not be mapped to protected classes.
- Synthetic group differences are designed into the generator and cannot establish fairness or effectiveness.
- SHAP describes model-score contributions; it is not a causal explanation or a justification for an employment decision.
- The four-fifths ratio is a screening heuristic, not a legal conclusion or fairness guarantee.
- HMAC IDs and a k-anonymity diagnostic are demonstrations; neither alone makes data anonymous or establishes LGPD compliance.

## Human oversight and rollback

No deployment is authorized or implemented. Any future real-world evaluation would require a separately approved purpose, privacy/legal review, validated job-related labels, representative data, human decision authority, appeal and incident processes, and documented acceptance criteria. If those controls failed, the model would be removed from the decision path and the documented non-model process restored; see [GOVERNANCE.md](./GOVERNANCE.md).
