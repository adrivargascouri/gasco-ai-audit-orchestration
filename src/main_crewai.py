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
from modular.data.csv_validator import validate_financial_data
from modular.hitl.final_approval import run_final_approval_workflow


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
    parser.add_argument(
        "--financial-file",
        type=Path,
        help=(
            "Optional company financial data CSV used for deterministic risk "
            "discovery."
        ),
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
    if args.financial_file:
        validations.append(
            (
                "financial",
                args.financial_file,
                validate_financial_data(args.financial_file),
            )
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
    financial_file: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Adapt official pipeline dataframes to the risk discovery agent input shape."""
    financial_data, financial_status = _aligned_financial_data(
        group_df,
        financial_file,
    )

    findings = pd.DataFrame({
        "entity_name": findings_df.get("Entity", pd.Series(dtype=object)),
        "finding_description": findings_df.get("Finding", pd.Series(dtype=object)),
        "severity": findings_df.get("Severity", pd.Series(dtype=object)),
    })

    group_entities = pd.DataFrame({
        "entity_name": group_df.get("Entity", pd.Series(dtype=object)),
        "manual_risk_flag": group_df.get(
            "manual_risk_flag",
            pd.Series(pd.NA, index=group_df.index),
        ),
    })
    if "manual_risk_flag" in financial_data.columns:
        financial_flags = (
            financial_data[["entity_name", "manual_risk_flag"]]
            .dropna(subset=["manual_risk_flag"])
            .drop_duplicates(subset=["entity_name"])
        )
        group_entities = group_entities.merge(
            financial_flags,
            on="entity_name",
            how="left",
            suffixes=("", "_financial"),
        )
        group_entities["manual_risk_flag"] = group_entities[
            "manual_risk_flag_financial"
        ].combine_first(group_entities["manual_risk_flag"])
        group_entities = group_entities.drop(columns=["manual_risk_flag_financial"])

    return {
        "financial_data": financial_data,
        "findings": findings,
        "group_entities": group_entities,
        "_financial_status": financial_status,
    }


def _clean_csv_dataframe(path: Path) -> pd.DataFrame:
    """Load a validated CSV and strip whitespace from labels and text cells."""
    dataframe = pd.read_csv(path, encoding="utf-8-sig")
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    string_columns = dataframe.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        dataframe[column] = dataframe[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return dataframe


def _entity_key(values: pd.Series) -> pd.Series:
    """Normalize entity labels for component-name alignment."""
    return values.astype(str).str.strip().str.casefold()


def _base_financial_data(group_df: pd.DataFrame) -> pd.DataFrame:
    """Build the group-derived financial frame used when no financial CSV is used."""
    total_assets = pd.to_numeric(
        group_df.get("Assets", pd.Series(dtype=float)),
        errors="coerce",
    )
    financial_data = pd.DataFrame({
        "entity_name": group_df.get("Entity", pd.Series(dtype=object)),
        "total_assets": total_assets,
    })
    total_group_assets = total_assets.sum()
    financial_data["asset_percentage"] = (
        total_assets / total_group_assets if total_group_assets > 0 else pd.NA
    )
    return financial_data


def _aligned_financial_data(
    group_df: pd.DataFrame,
    financial_file: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Align optional financial intake rows to official group entities."""
    base_financial_data = _base_financial_data(group_df)
    if financial_file is None:
        return base_financial_data, {
            "provided": False,
            "used": False,
            "message": (
                "Financial data: no --financial-file provided; risk discovery "
                "used group assets only."
            ),
        }

    financial_df = _clean_csv_dataframe(financial_file)
    financial_df["_entity_key"] = _entity_key(financial_df["entity_name"])

    aligned = base_financial_data.copy()
    aligned["_entity_key"] = _entity_key(aligned["entity_name"])
    aligned = aligned.merge(
        financial_df,
        on="_entity_key",
        how="left",
        suffixes=("", "_financial"),
    )

    matched_mask = aligned.get("entity_name_financial", pd.Series(dtype=object)).notna()
    matched_rows = int(matched_mask.sum())
    matched_entities = int(aligned.loc[matched_mask, "_entity_key"].nunique())
    group_keys = set(aligned["_entity_key"].dropna())
    ignored_rows = int((~financial_df["_entity_key"].isin(group_keys)).sum())

    if matched_rows == 0:
        return base_financial_data, {
            "provided": True,
            "used": False,
            "path": financial_file,
            "matched_rows": 0,
            "matched_entities": 0,
            "ignored_rows": ignored_rows,
            "message": (
                f"Financial data: provided at {financial_file}, but no entity "
                "names matched the loaded group structure; risk discovery used "
                "group assets only."
            ),
        }

    aligned["total_assets"] = pd.to_numeric(
        aligned.get("total_assets_financial", aligned["total_assets"]),
        errors="coerce",
    ).fillna(pd.to_numeric(aligned["total_assets"], errors="coerce"))
    total_group_assets = pd.to_numeric(base_financial_data["total_assets"], errors="coerce").sum()
    aligned["asset_percentage"] = (
        aligned["total_assets"] / total_group_assets if total_group_assets > 0 else pd.NA
    )

    drop_columns = [
        "_entity_key",
        "entity_name_financial",
        "total_assets_financial",
    ]
    aligned = aligned.drop(columns=[column for column in drop_columns if column in aligned])

    missing_group_entities = int(len(group_df) - matched_entities)
    return aligned, {
        "provided": True,
        "used": True,
        "path": financial_file,
        "matched_rows": matched_rows,
        "matched_entities": matched_entities,
        "ignored_rows": ignored_rows,
        "missing_group_entities": missing_group_entities,
        "message": (
            f"Financial data: provided and used from {financial_file}; "
            f"matched {matched_rows} financial row(s) across "
            f"{matched_entities} group entity/entities."
        ),
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
    financial_file: Path | None = None,
) -> tuple[dict[str, Path], dict[str, object]]:
    """Run deterministic risk discovery and export review-ready artifacts."""
    if audit_crew.group_df is None or audit_crew.findings_df is None:
        raise RuntimeError("Risk discovery requires loaded group and findings data")

    client_data = _risk_discovery_input(
        audit_crew.group_df,
        audit_crew.findings_df,
        financial_file=financial_file,
    )
    financial_status = client_data.pop("_financial_status")
    identified_risks = discover_risks(client_data)
    risk_workpaper = _build_risk_review_workpaper(identified_risks)

    identified_risks_path = output_directory / "identified_risks.csv"
    risk_workpaper_path = output_directory / "risk_review_workpaper.csv"
    identified_risks.to_csv(identified_risks_path, index=False)
    risk_workpaper.to_csv(risk_workpaper_path, index=False)

    return (
        {
            "identified_risks": identified_risks_path,
            "risk_review_workpaper": risk_workpaper_path,
        },
        financial_status,
    )


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
    risk_discovery_paths, financial_status = _export_risk_discovery_outputs(
        audit_crew,
        Path(config.output_directory),
        args.financial_file,
    )
    final_approval_paths = run_final_approval_workflow(config.output_directory)
    results["export_paths"].update(risk_discovery_paths)
    results["export_paths"].update(final_approval_paths)

    risk_result = results["risk_result"]
    coverage_result = results["coverage_result"]
    instruction_result = results["instruction_result"]

    print("\n==============================")
    print("GASCO CREWAI ML AUDIT WORKFLOW")
    print("==============================")
    print(f"\n{financial_status['message']}")
    if financial_status.get("used") and financial_status.get("ignored_rows"):
        print(
            "Financial data warning: "
            f"{financial_status['ignored_rows']} unmatched financial row(s) "
            "were ignored."
        )
    if financial_status.get("used") and financial_status.get("missing_group_entities"):
        print(
            "Financial data note: "
            f"{financial_status['missing_group_entities']} group entity/entities "
            "had no matching financial row."
        )

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
    print(
        "Final approved scope saved to: "
        f"{final_approval_paths['final_approved_scope']}"
    )
    print(f"Auditor feedback saved to: {final_approval_paths['auditor_feedback']}")


if __name__ == "__main__":
    main()
