"""ML-based modular entry point for audit scope orchestration."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from modular.config.loader import ConfigLoader
from modular.coverage.calculator import CoverageCalculator
from modular.data.loader import DataLoader
from modular.export.csv_exporter import CSVExporter
from modular.instructions.generator import InstructionGenerator
from modular.model.explainability import ScopeModelExplainability
from modular.model.model_registry import SCOPE_FEATURE_COLUMNS
from modular.scope_engine.ml_engine import MLScopeEngine


class MLAuditScopePipeline:
    """Pipeline variant that uses the local ML scope engine."""

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader()
        self.scope_engine = MLScopeEngine()
        self.coverage_calculator = CoverageCalculator(config.coverage, self.scope_engine)
        self.instruction_generator = InstructionGenerator(config.audit, self.scope_engine)
        self.csv_exporter = CSVExporter(config.output_directory)
        self.explainability = ScopeModelExplainability()

    def run(self) -> dict:
        group_df, findings_df = self.data_loader.load_group_data(
            self.config.group_data_path,
            self.config.findings_data_path,
        )
        scoped_df = self.scope_engine.recommend_scope(group_df)
        coverage_results = self.coverage_calculator.calculate_coverage(group_df)
        instruction_df = self.instruction_generator.generate_instructions(group_df)

        export_paths = {
            "scopes": self.csv_exporter.export_scope_recommendations(scoped_df),
            "instructions": self.csv_exporter.export_instructions(instruction_df),
            "coverage": self.csv_exporter.export_coverage_summary(coverage_results),
            "feature_importance": self.explainability.save_feature_importance(
                Path(self.config.output_directory) / "feature_importance.csv"
            ),
            "prediction_explanations": self.explainability.save_prediction_explanations(
                self.scope_engine.last_feature_frame,
                ["USA_Sub", "Germany_Sub", "Brazil_Sub"],
                Path(self.config.output_directory) / "prediction_explanations.txt",
            ),
        }

        return {
            "scoped_df": scoped_df,
            "coverage_results": coverage_results,
            "instruction_df": instruction_df,
            "export_paths": export_paths,
            "findings_df": findings_df,
            "feature_df": self.scope_engine.last_feature_frame.copy(),
        }

    def print_results(self, results: dict) -> None:
        scoped_df = results["scoped_df"]
        coverage_results = results["coverage_results"]
        instruction_df = results["instruction_df"]

        print("\n==============================")
        print("GASCO GROUP AUDIT ORCHESTRATOR - ML")
        print("==============================")

        print("\n1. ML SIGNIFICANCE & SCOPE RECOMMENDATION")
        print(scoped_df[[
            "Entity",
            "Country",
            "Assets",
            "Risk_Level",
            "Asset_Percentage",
            "Recommended_Scope",
            "Prediction_Confidence",
        ]])

        print("\n2. GROUP AUDIT COVERAGE")
        print(f"Total Group Assets: ${coverage_results['Total_Assets']:,}")
        print(f"Covered Assets: ${coverage_results['Covered_Assets']:,}")
        print(f"Coverage Percentage: {coverage_results['Coverage_Percentage']:.2f}%")
        print(f"Status: {coverage_results['Status']}")

        print("\n3. GROUP AUDIT INSTRUCTIONS")
        pd.set_option("display.max_colwidth", None)
        print(instruction_df[["Entity", "Country", "Recommended_Scope", "Audit_Instruction"]])

        print("\n4. DEBUG FEATURE VECTORS")
        debug_df = results["feature_df"]
        debug_df = debug_df[debug_df["Entity"].isin(["USA_Sub", "Germany_Sub"])]
        print(debug_df[["Entity", *SCOPE_FEATURE_COLUMNS]].to_string(index=False))

        print("\n5. TOP MODEL FEATURE IMPORTANCES")
        importance_path = results["export_paths"]["feature_importance"]
        importance_df = pd.read_csv(importance_path)
        print(importance_df.head(10).to_string(index=False))

        print("\n\nExported files:")
        for name, path in results["export_paths"].items():
            print(f"  {name}: {path}")


def main() -> None:
    config_path = Path(__file__).parent.parent / "modular" / "data" / "config.yaml"
    config = ConfigLoader.load_from_yaml(config_path)
    config = config.model_copy(update={"output_directory": "outputs_ml"})

    pipeline = MLAuditScopePipeline(config)
    results = pipeline.run()
    pipeline.print_results(results)


if __name__ == "__main__":
    main()
