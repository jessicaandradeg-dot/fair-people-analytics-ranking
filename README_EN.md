# Fair People Analytics Ranking (Synthetic Demo)

An educational portfolio prototype for grouped ranking, fairness measurement, explanations, drift triage, and pilot design. It is **not** a hiring system.

## Included methods

- Vacancy-group holdout: logistic reference vs. LightGBM LambdaRank.
- Macro-averaged Precision@k, Recall@k, and NDCG@k by vacancy.
- Group selection rates and the four-fifths heuristic, clearly not a legal test.
- Linear and SHAP explanations, PSI drift diagnostics, and a simulated randomized pilot.
- HMAC-SHA256 pseudonyms and a k-anonymity diagnostic; neither is anonymization.

All profiles, labels, groups, vacancies, and outcomes are synthetic. The project does not establish hiring validity, fairness, legal compliance, or LGPD compliance. See [MODEL_CARD.md](./MODEL_CARD.md) and [GOVERNANCE.md](./GOVERNANCE.md) for intended-use limits and rollback considerations.

Run locally on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
python -m pytest
python -m people_analytics
```
