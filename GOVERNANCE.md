# Learning Governance Checklist

This checklist documents topics the prototype deliberately does **not** resolve. It is a learning aid, not legal advice, a formal privacy impact assessment, or evidence of compliance with the LGPD or any other law.

## 1. Purpose and data minimization

- Use only the synthetic data in this repository for the portfolio demonstration.
- Before any real-data research, define and approve a specific purpose, lawful basis, necessity, retention, access, and deletion process with qualified organizational and legal/privacy reviewers.
- Do not collect fields simply because they are available. Document why each feature is necessary and job-related.
- Keep audit-only group data separate from model features; protect it with strict access controls if its use is ever authorized.
- Do not put personal data, credentials, private datasets, or confidential employer material in Git history, notebooks, logs, or generated reports.

## 2. Model and decision documentation

For every model or policy version, record:

- intended and prohibited uses;
- data sources, transformations, label definition, known coverage gaps, and limitations;
- feature list and rationale for each feature;
- training/evaluation split, random seed, metrics, group sample sizes, and uncertainty;
- fairness definition and why it is appropriate for the specific context;
- model/policy version, change date, reviewer, approval, and rollback instructions;
- documented human review, explanation, appeal, and incident-handling process.
- documented rollback trigger, decision owner, restoration method, and post-rollback review.

The current CLI reports its random seed, synthetic dataset size, model metrics, group audits, policy examples, and drift triage. It does not implement approvals, access control, an audit log service, case management, notices, or appeals.

## 3. Fairness review

- Audit relevant groups only where collection and use of those attributes are appropriate and authorized.
- Review sample size, missingness, intersectional groups, label quality, and uncertainty before comparing rates.
- Do not assume that excluding protected attributes eliminates proxy discrimination.
- Do not assume that equal selection rates are the sole or universally correct fairness objective.
- Review the effect of any mitigation on ranking quality, false negatives, accessibility, affected people, and legal/ethical obligations.

This repository's group labels and outcomes are invented. The synthetic race-like categories have no mapping to real identities.

## 4. Drift and operations

- Define reference populations, feature schemas, data-quality checks, minimum sample sizes, PSI thresholds, and accountable reviewers before deployment.
- Investigate changes in feature distributions; a drift score does not prove model failure or tell an operator what intervention is appropriate.
- Monitor target quality, calibration, performance, fairness, and operational incidents separately when valid labels become available.
- Document alert ownership, escalation, rollback, revalidation, and communication to affected stakeholders.

The repository contains offline distribution comparisons only. It has no live monitoring, alerting, production data, service-level objectives, or automated rollback mechanism.

### Rollback plan (design checklist only)

Before any authorized real-world pilot, name a responsible owner and define measurable stop conditions for data quality, model quality, group-impact review, privacy/security incidents, and operational failure. On a stop condition, disable the model output, restore the previously approved non-model process, preserve only authorized audit evidence, notify the designated stakeholders, investigate root cause, and require documented re-validation and approval before any re-entry. This repository implements none of those controls; the steps are a planning template, not an operational rollback capability.

## 5. Controlled evaluation

The repository's A/B code is a simulation for practicing randomization, outcome definitions, and uncertainty intervals. A real pilot would need separate organizational authorization, a reviewed protocol, privacy/security safeguards, approved participant communication, appropriate human oversight, and valid outcome/effort measures. No real pilot has been conducted.
