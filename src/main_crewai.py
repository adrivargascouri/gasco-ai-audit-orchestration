"""Current primary GASCO pipeline: the CrewAI-orchestrated ML audit workflow."""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

sys.path.insert(0, str(Path(__file__).parent.parent))

from modular.config.loader import ConfigLoader
from modular.crew.audit_crew import GASCOAuditCrew
from modular.data.company_csv_validator import (
    validate_findings_csv,
    validate_group_structure_csv,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GASCO CrewAI audit scoping pipeline."
    )
    parser.add_argument(
        "--group-file",
        type=Path,
        help="Company group structure CSV (uses the default JSON when omitted).",
    )
    parser.add_argument(
        "--findings-file",
        type=Path,
        help="Optional company findings CSV (uses the default findings when omitted).",
    )
    args = parser.parse_args()

    validations = []
    if args.group_file:
        validations.append(
            ("group", args.group_file, validate_group_structure_csv(args.group_file))
        )
    if args.findings_file:
        validations.append(
            ("findings", args.findings_file, validate_findings_csv(args.findings_file))
        )

    errors = []
    for label, path, result in validations:
        for error in result["errors"]:
            errors.append(f"{label} file '{path}': {error}")
    if errors:
        parser.error("Company CSV validation failed:\n  - " + "\n  - ".join(errors))
    return args


def main() -> None:
    args = _parse_args()
    config_path = Path(__file__).parent.parent / "modular" / "data" / "config.yaml"
    config = ConfigLoader.load_from_yaml(config_path)
    config_updates = {"output_directory": "outputs_crewai"}
    if args.group_file:
        config_updates["group_data_path"] = str(args.group_file)
    if args.findings_file:
        config_updates["findings_data_path"] = str(args.findings_file)
    config = config.model_copy(update=config_updates)

    audit_crew = GASCOAuditCrew(config)
    results = audit_crew.run()

    risk_result = results["risk_result"]
    coverage_result = results["coverage_result"]
    instruction_result = results["instruction_result"]

    print("\n==============================")
    print("GASCO CREWAI ML AUDIT WORKFLOW")
    print("==============================")

    print("\n1. CREWAI WORKFLOW SUMMARY")
    print(results["agent_outputs"])

    print("\n2. ML SCOPE RECOMMENDATIONS")
    print(risk_result.scoped_df[[
        "Entity",
        "Country",
        "Assets",
        "Risk_Level",
        "Asset_Percentage",
        "Original_ML_Scope",
        "Guardrail_Adjusted_Scope",
        "Guardrail_Action",
        "Requires_Human_Review",
        "Prediction_Confidence",
    ]].to_string(index=False))

    print("\n3. COVERAGE")
    print(pd.DataFrame([coverage_result.summary]).to_string(index=False))

    print("\n4. SAMPLE EXPLANATIONS")
    preferred_entities = ["USA_Sub", "Germany_Sub", "Brazil_Sub"]
    sample_entities = [
        entity for entity in preferred_entities if entity in risk_result.explanations
    ]
    sample_entities.extend(
        entity
        for entity in risk_result.explanations
        if entity not in sample_entities
    )
    for entity in sample_entities[:3]:
        print()
        print(risk_result.explanations[entity])

    print("\n5. SAMPLE AUDIT INSTRUCTIONS")
    print(instruction_result.instruction_df[[
        "Entity",
        "Original_ML_Scope",
        "Guardrail_Adjusted_Scope",
        "Prediction_Confidence",
        "Requires_Human_Review",
        "ML_Explanation_Summary",
    ]].head(5).to_string(index=False))

    print("\n\nExported files:")
    for name, path in results["export_paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
