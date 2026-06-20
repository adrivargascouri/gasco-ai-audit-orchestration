"""Current primary GASCO pipeline: the CrewAI-orchestrated ML audit workflow."""
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


def main() -> None:
    config_path = Path(__file__).parent.parent / "modular" / "data" / "config.yaml"
    config = ConfigLoader.load_from_yaml(config_path)
    config = config.model_copy(update={"output_directory": "outputs_crewai"})

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
    for entity in ["USA_Sub", "Germany_Sub", "Brazil_Sub"]:
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
