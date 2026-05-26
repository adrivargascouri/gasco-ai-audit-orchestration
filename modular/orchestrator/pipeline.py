"""Main orchestrator pipeline for group audit processing."""
import pandas as pd
from pathlib import Path
from ..config.config_schema import AppConfig
from ..data.loader import DataLoader
from ..scope_engine.rules_engine import RulesScopeEngine
from ..coverage.calculator import CoverageCalculator
from ..instructions.generator import InstructionGenerator
from ..export.csv_exporter import CSVExporter


class AuditScopePipeline:
    """Orchestrates the complete audit scope analysis pipeline.

    Pipeline stages:
    1. Load data (group structure and findings)
    2. Recommend scopes (using rules-based engine)
    3. Calculate coverage (for the group)
    4. Generate instructions (audit procedures for each entity)
    5. Export results (to CSV files)
    """

    def __init__(self, config: AppConfig):
        """Initialize the audit scope pipeline.

        Args:
            config: AppConfig with all settings
        """
        self.config = config

        # Initialize components
        self.data_loader = DataLoader()
        self.scope_engine = RulesScopeEngine(config.scope_rules)
        self.coverage_calculator = CoverageCalculator(config.coverage, self.scope_engine)
        self.instruction_generator = InstructionGenerator(config.audit, self.scope_engine)
        self.csv_exporter = CSVExporter(config.output_directory)

    def run(self) -> dict:
        """Execute the complete audit scope pipeline.

        Returns:
            Dict with results:
            - scoped_df: DataFrame with scope recommendations
            - coverage_results: Dict with coverage metrics
            - instruction_df: DataFrame with audit instructions
            - export_paths: Dict with paths to exported CSV files
        """
        # 1. Load data
        group_df, findings_df = self.data_loader.load_group_data(
            self.config.group_data_path,
            self.config.findings_data_path
        )

        # 2. Recommend scopes
        scoped_df = self.scope_engine.recommend_scope(group_df)

        # 3. Calculate coverage
        coverage_results = self.coverage_calculator.calculate_coverage(group_df)

        # 4. Generate instructions
        instruction_df = self.instruction_generator.generate_instructions(group_df)

        # 5. Export results
        export_paths = {
            "scopes": self.csv_exporter.export_scope_recommendations(scoped_df),
            "instructions": self.csv_exporter.export_instructions(instruction_df),
            "coverage": self.csv_exporter.export_coverage_summary(coverage_results)
        }

        return {
            "scoped_df": scoped_df,
            "coverage_results": coverage_results,
            "instruction_df": instruction_df,
            "export_paths": export_paths,
            "findings_df": findings_df
        }

    def print_results(self, results: dict) -> None:
        """Print pipeline results to console.

        Args:
            results: Dict returned from run()
        """
        scoped_df = results["scoped_df"]
        coverage_results = results["coverage_results"]
        instruction_df = results["instruction_df"]

        print("\n==============================")
        print("GASCO GROUP AUDIT ORCHESTRATOR")
        print("==============================")

        print("\n1. SIGNIFICANCE & SCOPE RECOMMENDATION")
        print(scoped_df[["Entity", "Country", "Assets", "Risk_Level", "Recommended_Scope"]])

        print("\n2. GROUP AUDIT COVERAGE")
        print(f"Total Group Assets: ${coverage_results['Total_Assets']:,}")
        print(f"Covered Assets: ${coverage_results['Covered_Assets']:,}")
        print(f"Coverage Percentage: {coverage_results['Coverage_Percentage']:.2f}%")
        print(f"Status: {coverage_results['Status']}")

        print("\n3. GROUP AUDIT INSTRUCTIONS")
        pd.set_option("display.max_colwidth", None)
        print(instruction_df[["Entity", "Country", "Recommended_Scope", "Audit_Instruction"]])
