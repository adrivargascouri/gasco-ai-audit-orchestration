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

The CSV files under `templates/` support a separate client-data intake workflow;
`src/main_crewai.py` does not currently read them.

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
| `prediction_explanations.txt` | Selected human-readable ML prediction explanations |
| `crew_workflow_summary.txt` | Deterministic CrewAI stage summary |
| `bdo_documentation_memo.txt` | Methodology-oriented scoping and coverage memo |
| `component_auditor_instruction_pack.csv` | Component auditor instruction pack |
| `human_review_ai_recommendations.csv` | AI recommendations and evidence prepared for auditor review |
| `auditor_review_workpaper.csv` | Blank/pending auditor decision template |
| `audit_trail.csv` | Initial audit-trail rows with pending final-decision fields |
| `final_human_review_report.txt` | HITL status report and completion guidance |

The pipeline does not clear an output directory before running. A file present in
`outputs_crewai/` but not listed above may be an artifact from another or older
workflow and should not be attributed to the latest primary run without checking
its provenance.

## Current Limitations

- CrewAI orchestration is deterministic and sequential.
- `LocalCrewLLM` is a local stub/shim used to satisfy CrewAI orchestration; it does
  not perform generative reasoning or call an external LLM.
- HITL currently generates review templates and pending audit-trail fields, but it
  does not ingest completed auditor decisions back into the pipeline.
- Older rule-based, modular comparison, and standalone ML pipelines remain for
  historical comparison and may produce different output families.
- Automated tests are not yet implemented.
