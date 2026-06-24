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
from modular.agents.risk_discovery_agent import discover_risks
from modular.crew.audit_crew import GASCOAuditCrew
from modular.data.company_csv_validator import (
    validate_findings_csv,
    validate_group_structure_csv,
)


RISK_REVIEW_COLUMNS = [
    "entity_name",
    "risk_type",
    "risk_description",
    "severity",
    "source",
    "evidence_value",
    "confidence",
    "requires_human_review",
    "auditor_comment",
    "review_status",
]


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


def _risk_discovery_input(
    group_df: pd.DataFrame,
    findings_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Adapt official pipeline dataframes to the risk discovery agent input shape."""
    financial_data = pd.DataFrame({
        "entity_name": group_df.get("Entity", pd.Series(dtype=object)),
        "total_assets": group_df.get("Assets", pd.Series(dtype=float)),
    })

    findings = pd.DataFrame({
        "entity_name": findings_df.get("Entity", pd.Series(dtype=object)),
        "finding_description": findings_df.get("Finding", pd.Series(dtype=object)),
        "severity": findings_df.get("Severity", pd.Series(dtype=object)),
    })

    group_entities = pd.DataFrame({
        "entity_name": group_df.get("Entity", pd.Series(dtype=object)),
        "manual_risk_flag": group_df.get("manual_risk_flag", pd.Series(dtype=object)),
    })

    return {
        "financial_data": financial_data,
        "findings": findings,
        "group_entities": group_entities,
    }


def _build_risk_review_workpaper(identified_risks: pd.DataFrame) -> pd.DataFrame:
    """Create the auditor review workpaper from discovered risks."""
    workpaper = identified_risks.copy()
    confidence = pd.to_numeric(workpaper["confidence"], errors="coerce")
    requires_human_review = (
        workpaper["severity"].astype(str).str.strip().str.casefold().isin(
            {"high", "critical"}
        )
        | confidence.lt(0.70)
        | workpaper["risk_type"].astype(str).str.contains(
            "Significant Component Risk",
            case=False,
            na=False,
        )
    )

    workpaper["requires_human_review"] = requires_human_review
    workpaper["auditor_comment"] = ""
    workpaper["review_status"] = requires_human_review.map({
        True: "Pending",
        False: "Not Required",
    })

    return workpaper[RISK_REVIEW_COLUMNS]


def _export_risk_discovery_outputs(
    audit_crew: GASCOAuditCrew,
    output_directory: Path,
) -> dict[str, Path]:
    """Run deterministic risk discovery and export review-ready artifacts."""
    if audit_crew.group_df is None or audit_crew.findings_df is None:
        raise RuntimeError("Risk discovery requires loaded group and findings data")

    client_data = _risk_discovery_input(audit_crew.group_df, audit_crew.findings_df)
    identified_risks = discover_risks(client_data)
    risk_workpaper = _build_risk_review_workpaper(identified_risks)

    identified_risks_path = output_directory / "identified_risks.csv"
    risk_workpaper_path = output_directory / "risk_review_workpaper.csv"
    identified_risks.to_csv(identified_risks_path, index=False)
    risk_workpaper.to_csv(risk_workpaper_path, index=False)

    return {
        "identified_risks": identified_risks_path,
        "risk_review_workpaper": risk_workpaper_path,
    }


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
    risk_discovery_paths = _export_risk_discovery_outputs(
        audit_crew,
        Path(config.output_directory),
    )
    results["export_paths"].update(risk_discovery_paths)

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
    print(f"Identified risks saved to: {risk_discovery_paths['identified_risks']}")
    print(
        "Risk review workpaper saved to: "
        f"{risk_discovery_paths['risk_review_workpaper']}"
    )


if __name__ == "__main__":
    main()
