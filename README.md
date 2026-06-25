# GASCO Audit Scoping

GASCO is an audit-scoping prototype that combines a local machine-learning scope
classifier, deterministic CrewAI orchestration, BDO-style methodology guardrails,
coverage analysis, component instructions, and human-review workpapers. It is an
auditor decision-support system: its recommendations are not final audit decisions.

## Official Pipeline

The current primary execution path is `src/main_crewai.py`. It runs the following
stages sequentially:

1. Load configuration, group entity data, findings, ML reference data, and the
   persisted scope model.
2. Generate local ML scope predictions and explanations.
3. Apply the existing methodology guardrails and identify recommendations that
   require human review.
4. Calculate group coverage and identify risky uncovered entities.
5. Generate group audit instructions and supporting documentation.
6. Export CrewAI workflow, explainability, guardrail, and HITL review artifacts to
   `outputs_crewai/`.

The CrewAI agents coordinate these deterministic local stages. They do not call an
external generative model.

Other entry points remain in the repository for comparison and historical use:

| Entry point | Status | Purpose |
| --- | --- | --- |
| `src/main_crewai.py` | **Current primary** | CrewAI-orchestrated ML pipeline with guardrails and HITL artifacts |
| `src/main.py` | Legacy | Original rule-based pipeline |
| `src/main_v2.py` | Comparison | Modular rule-based parity/comparison pipeline |
| `src/main_ml.py` | Standalone | ML-only pipeline without the current CrewAI workflow |

## Setup

Run commands from the repository root because several configured data paths are
relative to that directory.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The direct runtime dependencies are `pandas`, `PyYAML`, `pydantic`,
`scikit-learn`, `joblib`, `crewai`, and `numpy`. See `requirements.txt` for the
recorded versions and supporting packages.

## Run

Run the official pipeline from the repository root:

```powershell
python src/main_crewai.py
```

The command prints a workflow summary, scope recommendations, coverage, sample
explanations, sample instructions, and the paths of generated files. It also
creates `outputs_crewai/` if it does not exist and overwrites same-named current
artifacts there.

Historical entry points can be executed separately when comparison is required:

```powershell
python src/main.py
python src/main_v2.py
python src/main_ml.py
```

These are not the official production path.

## Inputs

The primary pipeline uses these repository files:

| Path | Purpose |
| --- | --- |
| `modular/data/config.yaml` | Scope, coverage, instruction, guardrail, data-path, and output defaults; the primary entry point overrides the output directory to `outputs_crewai/` |
| `data/group_structure.json` | Current group entities and financial/risk attributes |
| `data/findings_repo.csv` | Entity-level historical findings used in scoping features |
| `data/processed/training_dataset.csv` | Reference training data used for feature validation, scaling, estimation, and explainability |
| `data/raw/historical_audit_data.csv` | Historical country-risk reference data; the current engine falls back to defaults if it is absent |
| `models/scope_model.pkl` | Persisted local scope classification model used for inference |

## Using GASCO with company-provided CSV data

Company users can run the official pipeline with CSV files only; no programming
or JSON editing is required. Start with the files in `data/templates/`:

- `company_group_structure_template.csv`
- `company_findings_template.csv`
- `company_financial_data_template.csv`

Keep the header names unchanged, replace the example rows with company data, and
save the completed files in `data/client_uploads/`. Do not commit real client data.

The group structure file requires `entity_name`, `country`, `total_assets`,
`revenue`, and `risk_level`. Assets and revenue must be non-negative numbers.
The findings file requires `entity_name`, `finding_description`, `severity`, and
`year`; `year` may be blank, but must be a whole number when provided. Accepted
values for both `risk_level` and `severity` are `Low`, `Medium`, `High`, and
`Critical`. The financial data file requires `entity_name`, `year`,
`current_assets`, `current_liabilities`, `total_assets`, `total_revenue`,
`net_income`, `liquidity_ratio`, and `debt_ratio`; entity names are matched to
the group structure by component/entity name. It may also include
`manual_risk_flag`.

Run GASCO from the repository root with completed files:

```powershell
python src/main_crewai.py --group-file data/client_uploads/company_group_structure.csv --findings-file data/client_uploads/company_findings.csv --financial-file data/client_uploads/company_financial_data.csv
```

The findings and financial files are optional. Omit `--findings-file` to use the
existing default findings repository. Omit `--financial-file` to use group assets
only for deterministic risk discovery. Omitting all three arguments preserves the
existing default run. Invalid files are rejected before the pipeline starts, with
row-specific messages that explain what needs to be corrected.

## Outputs

Output directories have distinct status and should not be treated as equivalent:

| Directory | Status and meaning |
| --- | --- |
| `outputs_crewai/` | **Current primary outputs** from `src/main_crewai.py` |
| `outputs_ml/` | Historical/standalone ML-only outputs from `src/main_ml.py` |
| `outputs/` | Legacy original rule-based outputs; also the default configured directory used by the modular comparison path |
| `outputs_v1/` | Retained legacy/comparison output snapshot |
| `outputs_v2/` | Retained legacy/comparison output snapshot |

The primary run generates these artifacts in `outputs_crewai/`:

| Artifact | Meaning |
| --- | --- |
| `significance_scope_recommendation.csv` | ML recommendations, guardrail adjustments, review flags, and methodology fields by component |
| `coverage_summary.csv` | Group asset coverage result for included scopes |
| `group_audit_instructions.csv` | Component-level audit instructions based on adjusted scope |
| `risky_uncovered_entities.csv` | Risky components outside the scopes included in coverage |
| `feature_importance.csv` | Feature importance from the persisted classifier |
| `identified_risks.csv` | Deterministic risks discovered from the same group, findings, and optional financial data used by the official run |
| `risk_review_workpaper.csv` | Auditor review workpaper for discovered risks, with pending status for high-severity, low-confidence, or significant-component risks |
| `prediction_explanations.txt` | Selected human-readable ML prediction explanations |
| `crew_workflow_summary.txt` | Deterministic CrewAI stage summary |
| `bdo_documentation_memo.txt` | Methodology-oriented scoping and coverage memo |
| `component_auditor_instruction_pack.csv` | Component auditor instruction pack |
| `human_review_ai_recommendations.csv` | AI recommendations and evidence prepared for auditor review |
| `auditor_review_workpaper.csv` | Blank/pending auditor decision template |
| `auditor_feedback.csv` | Structured auditor decision labels derived from the auditor review workpaper for future ML improvement |
| `audit_trail.csv` | Initial audit-trail rows with pending final-decision fields |
| `final_human_review_report.txt` | HITL status report and completion guidance |

### Risk discovery review outputs

The official pipeline also runs the existing deterministic Risk Discovery Agent
after the CrewAI workflow has completed. When `--financial-file` is provided,
validated financial indicators are aligned to group entities and can produce
liquidity, debt, significant-component, and manual-flag risks.
`identified_risks.csv` preserves the raw discovered risk records.
`risk_review_workpaper.csv` adds auditor-review fields: high or critical risks,
low-confidence risks, and significant component risks are marked `Pending`; all
other discovered risks are marked `Not Required`.

### Auditor feedback loop

The official pipeline creates `auditor_feedback.csv` from
`auditor_review_workpaper.csv` on every run. It stores each entity's AI
recommended scope, final auditor scope, decision status, auditor comment,
feedback label, feedback reason, and whether the row is usable for future
training.

This file does not retrain the local ML model yet. It prepares labeled human
decisions so a future model-improvement or retraining workflow can distinguish
accepted AI recommendations from auditor overrides while leaving pending
decisions out of training data.

The pipeline does not clear an output directory before running. A file present in
`outputs_crewai/` but not listed above may be an artifact from another or older
workflow and should not be attributed to the latest primary run without checking
its provenance.

## Current Limitations

- CrewAI orchestration is deterministic and sequential.
- `LocalCrewLLM` is a local stub/shim used to satisfy CrewAI orchestration; it does
  not perform generative reasoning or call an external LLM.
- HITL currently generates review templates, pending audit-trail fields, and
  auditor feedback labels, but it does not retrain the ML model.
- Older rule-based, modular comparison, and standalone ML pipelines remain for
  historical comparison and may produce different output families.
- Automated tests are not yet implemented.
