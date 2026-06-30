# Model Card: GASCO Scope Prediction Model

Last reviewed: 2026-06-30

## 1. Model Overview

The GASCO scope prediction model is a local supervised machine-learning classifier used within the GASCO audit-scoping prototype. Its purpose is to recommend an audit scope category for group components based on quantitative financial, risk, findings, and entity-level features.

The model is used as an auditor decision-support tool. It provides recommendations, confidence scores, and explanatory signals; it does not make final audit scoping decisions. Final scoping remains subject to BDO-style methodology guardrails and human auditor review.

| Item | Description |
| --- | --- |
| Model name | `gasco_scope_random_forest` |
| Inference engine | `MLScopeEngine` |
| Engine version | `2.0-local` |
| Model version | `phase_2b_local` |
| Saved artifact | `models/scope_model.pkl` |
| Model type | Multi-class supervised classifier |
| Framework | scikit-learn RandomForestClassifier |
| Classes | `Analytical Procedures`, `Specific Procedures`, `Full Scope` |
| Primary pipeline | `src/main_crewai.py` |

No project training notebooks were found in the repository inspection. The training and feature engineering workflow is script-based, primarily through `modular/features/engineer.py` and `modular/model/training.py`.

## 2. Intended Use

The model is intended to support group audit scoping by recommending one of three scope categories for each component:

| Recommended scope | Meaning in GASCO workflow |
| --- | --- |
| `Analytical Procedures` | Lower-intensity analytical work recommendation |
| `Specific Procedures` | Targeted audit procedures for elevated risk or importance |
| `Full Scope` | More comprehensive component audit work |

Appropriate uses include:

- Producing preliminary audit scope recommendations for group components.
- Supporting consistency checks across components using a reproducible local model.
- Highlighting components that require review due to low confidence, high risk, materiality, financial-risk guardrails, or methodology constraints.
- Generating documentation artifacts, including recommendation explanations, guardrail reasons, auditor workpapers, and audit trails.

Inappropriate uses include:

- Treating model output as a final audit decision.
- Using the model without qualified auditor review.
- Using the model as evidence that audit standards have been satisfied.
- Applying the model to real client engagements without validating data quality, model fit, regulatory requirements, and human approval controls.

## 3. Model Architecture

The saved model is a `sklearn.ensemble.RandomForestClassifier` with the following recorded configuration:

| Parameter | Value |
| --- | --- |
| Number of trees | 300 |
| Random state | 42 |
| Class weighting | `balanced` |
| Training split | 75 percent train, 25 percent test |
| Test strategy | Stratified split |
| Target column | `target_scope` |

The model predicts encoded class labels, which are decoded as:

| Encoded class | Scope label |
| --- | --- |
| 0 | `Analytical Procedures` |
| 1 | `Specific Procedures` |
| 2 | `Full Scope` |

The CrewAI workflow in the current primary pipeline orchestrates deterministic local stages. The repository README states that the agents do not call an external generative model. The scope classifier itself is local and does not use an LLM.

The model layer is distinct from the BDO guardrail layer. The RandomForest model generates an initial scope recommendation and confidence score. BDO-style guardrails are applied after the ML prediction to adjust, document, or flag recommendations before coverage calculation, instruction generation, human review, final approval, and feedback export.

## 4. Input Features

The persisted model expects 11 numeric features in a fixed order:

| Feature | Description |
| --- | --- |
| `revenue` | Component revenue, scaled for model input using training-data implied totals during inference |
| `assets` | Component assets, scaled for model input using training-data implied totals during inference |
| `revenue_percentage` | Component revenue as a percentage of total group revenue |
| `assets_percentage` | Component assets as a percentage of total group assets |
| `risk_level_encoded` | Encoded risk level: Low=1, Medium=2, High=3, Critical=4 |
| `country_risk_score` | Country risk score from historical reference data, with fallback behavior when unavailable |
| `prior_findings_count` | Number of prior findings associated with the component |
| `severe_findings_count` | Count of high or critical findings associated with the component |
| `growth_rate` | Historical or estimated growth signal |
| `liquidity_ratio` | Historical or estimated liquidity signal |
| `manual_risk_flag_encoded` | Encoded manual risk flag: No=0, Yes=1 |

The official pipeline accepts group entity data containing entity name, country, revenue, assets, and risk level. Findings and optional financial data can also be provided. At inference time, the ML engine derives percentages from group totals, maps risk levels, looks up country risk where possible, counts findings, and estimates missing growth, liquidity, and manual-risk features using nearest neighbors in the processed training data. If the component risk level is high or severe findings are present, the manual risk flag feature can be forced to the positive class.

This feature estimation is useful for a prototype but should be treated as a limitation for production use because estimated features may not reflect the actual current-period economics of a component.

## 5. Training Data

The processed training file is `data/processed/training_dataset.csv`. It contains 225 rows, 11 model features, and the encoded target column. The raw source file inspected in the repository is `data/raw/historical_audit_data.csv`, also with 225 rows.

The data should be treated as synthetic or synthetic-augmented demonstration data. The repository includes `expand_data.py`, which explicitly expands the historical audit dataset by creating variations of existing rows, adding randomized perturbations to entity names and numeric features while preserving broad scope patterns. The repository inspection did not identify evidence that the full dataset represents verified real client audit history.

Processed training-data summary:

| Item | Value |
| --- | --- |
| Rows | 225 |
| Missing values | 0 |
| Target classes | 3 |
| Analytical Procedures | 102 rows |
| Specific Procedures | 93 rows |
| Full Scope | 30 rows |

The training data includes financial magnitude features, relative size features, risk levels, country-risk scores, prior findings, severe findings, growth, liquidity, manual risk flags, and target scope labels. Because the data is small and synthetic-augmented, the model should be viewed as a prototype baseline rather than a validated audit-scoping model.

## 6. Performance Metrics

The saved model artifact records the following held-out test performance from the training workflow:

| Metric | Value |
| --- | --- |
| Test rows | 57 |
| Overall accuracy | 0.9298 |
| Macro average precision | 0.95 |
| Macro average recall | 0.92 |
| Macro average F1-score | 0.93 |
| Weighted average precision | 0.94 |
| Weighted average recall | 0.93 |
| Weighted average F1-score | 0.93 |

Per-class test performance:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Analytical Procedures | 1.00 | 0.88 | 0.94 | 26 |
| Specific Procedures | 0.85 | 1.00 | 0.92 | 23 |
| Full Scope | 1.00 | 0.88 | 0.93 | 8 |

Feature importance from the current exported output ranks the largest model signals as:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `risk_level_encoded` | 0.2096 |
| 2 | `liquidity_ratio` | 0.1881 |
| 3 | `manual_risk_flag_encoded` | 0.1462 |
| 4 | `assets_percentage` | 0.1416 |
| 5 | `assets` | 0.0786 |

These metrics should not be interpreted as production-grade validation. They are based on a small synthetic-augmented dataset and a single train-test split. The repository does not provide independent external validation, calibration analysis, fairness analysis, stress testing across client types, or longitudinal drift monitoring.

The current `outputs_crewai/` artifacts appear to represent a small sample run with 3 components. In that run, all 3 components required human review, 2 recommendations were adjusted by guardrails, and guardrail-adjusted coverage was 100 percent. This is useful workflow evidence, not evidence of broad model generalization.

## 7. Limitations

Key limitations include:

- The data is synthetic or synthetic-augmented and should not be assumed to represent real-world audit populations.
- The dataset is small, with only 225 rows and only 30 `Full Scope` examples.
- The reported performance is from a single held-out split, not from external validation.
- The model has not been demonstrated to generalize across industries, jurisdictions, reporting frameworks, firm methodologies, or changing economic conditions.
- Some inference-time features are estimated from nearest training neighbors, which may introduce approximation error.
- Country risk scores and risk-level encodings may embed subjective or jurisdictional assumptions.
- Feature importance explanations are model-level and heuristic; they are not causal explanations.
- The model does not evaluate qualitative audit evidence, management integrity, fraud indicators, complex group structures, legal requirements, or engagement-specific professional judgment unless those are represented in structured inputs or guardrails.
- The HITL feedback export prepares labels for future improvement, but the current workflow does not automatically retrain the model from auditor feedback.
- If auditor review fields are left blank, the final approval workflow can default to the AI/guardrail recommendation with `final_decision_source=ai_default`; this should not be treated as actual human approval.

## 8. Human Oversight

GASCO is designed as a human-in-the-loop audit support workflow. The model recommends scope categories, but auditors are expected to review, modify, approve, or reject the recommendation before final scoping.

Human oversight is implemented through:

- `human_review_ai_recommendations.csv`, which presents the AI/guardrail recommendation, confidence, evidence, guardrail triggers, and review reasons.
- `auditor_review_workpaper.csv`, which gives auditors fields for final scope, decision status, comments, approver, and approval date.
- `audit_trail.csv`, which records the original ML recommendation, guardrail-adjusted recommendation, final auditor decision fields, timestamps, and reasons.
- `final_approved_scope.csv`, which records the final scope and whether the source was auditor-provided or an AI default.
- `auditor_feedback.csv`, which labels accepted, overridden, pending, or AI-default decisions for future model improvement.

BDO-style guardrails reduce model risk by enforcing deterministic review and minimum-scope policies after ML prediction. The inspected configuration and implementation include:

- Significant components at or above 15 percent of group assets are escalated toward `Full Scope` and partner review.
- High or critical risk levels require at least `Specific Procedures`.
- Severe prior findings prevent analytical-only treatment without review.
- Low ML confidence below 0.65 triggers human review.
- Guardrail-adjusted coverage is checked against a 70 percent minimum coverage threshold.
- Liquidity risk and high debt risk require at least `Specific Procedures`.
- Manual risk flags require human review.
- Multiple financial risks on the same component require human review.

These controls reduce the chance that a low-confidence or under-scoped recommendation passes through unnoticed. They do not eliminate the need for professional judgment, engagement-specific procedures, or audit partner approval.

## 9. Ethical Considerations

The model should be governed as an assistive system in a high-accountability professional context. Ethical considerations include:

- Avoiding automation bias: users must not treat a model recommendation as authoritative merely because it is generated by software.
- Preserving auditor accountability: the auditor, not the model, makes the final scoping decision.
- Protecting client confidentiality: real client financial data and findings should be handled under appropriate access, retention, and security controls.
- Monitoring bias: country-risk scores, manual risk flags, and risk-level labels may reflect subjective judgments or historical practices that require review.
- Maintaining transparency: recommendations should be accompanied by confidence scores, feature-based explanations, guardrail reasons, and audit-trail documentation.
- Preventing overclaiming: the current synthetic-augmented dataset and prototype workflow do not support claims of regulatory compliance, audit sufficiency, or production reliability.

## 10. Future Improvements

Recommended improvements before broader deployment include:

- Replace or supplement synthetic-augmented data with validated historical audit outcomes and auditor-approved final scopes.
- Add independent validation sets separated by client, year, industry, geography, and engagement type.
- Evaluate calibration of prediction probabilities and set empirically justified confidence thresholds.
- Add model monitoring for data drift, class drift, and changes in auditor override rates.
- Incorporate auditor feedback into a controlled retraining workflow with versioned datasets and approval gates.
- Expand explainability beyond feature importance, for example with local explanation methods and clearer counterfactual examples.
- Add fairness and bias review for country-risk, risk-level, and manual-flag features.
- Improve feature capture so liquidity, growth, and manual risk indicators come from current validated data rather than nearest-neighbor estimates where possible.
- Strengthen documentation of data provenance, labeling policy, and audit methodology alignment.
- Add model governance artifacts, including model owner, validation owner, approval date, monitoring cadence, and retirement criteria.

## Strengths

- Local, reproducible model with explicit feature order, labels, and saved artifact.
- Clear separation between ML prediction, deterministic BDO-style guardrails, and human auditor approval.
- Human-readable explanations and exported workpapers support reviewability.
- Guardrails catch low confidence, high risk, significant components, severe findings, and financial-risk triggers.
- Audit-trail and feedback artifacts create a path toward future supervised improvement.

## Risks

- The training data is synthetic or synthetic-augmented and not sufficient for production claims.
- The dataset is small and class support for `Full Scope` is limited.
- Reported accuracy may overstate performance on real engagements.
- Inference can rely on estimated features when current-period data is unavailable.
- Users may over-rely on recommendations unless HITL controls are enforced.
- AI-default final scopes are not equivalent to auditor-approved decisions.

## Deployment Readiness

Deployment readiness: limited prototype / controlled pilot only.

The model is suitable for demonstration, academic documentation, and controlled internal evaluation with human review. It is not ready for autonomous audit scoping or unsupervised production use. Before production deployment, GASCO would need validated real-world training data, independent model validation, documented audit methodology approval, security and privacy review, monitoring controls, and enforced human approval before any recommendation is treated as final.
