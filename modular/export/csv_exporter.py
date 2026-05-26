"""CSV export functionality."""
import os
from pathlib import Path
import pandas as pd


class CSVExporter:
    """Exports audit results to CSV files.

    Handles writing scope recommendations, audit instructions, and
    coverage summaries to CSV format.
    """

    def __init__(self, output_directory: str = "outputs"):
        """Initialize CSV exporter.

        Args:
            output_directory: Directory to write CSV files to
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def export_scope_recommendations(
        self,
        scoped_df: pd.DataFrame,
        filename: str = "significance_scope_recommendation.csv"
    ) -> Path:
        """Export scope recommendations to CSV.

        Args:
            scoped_df: DataFrame with scope recommendations
            filename: Output filename

        Returns:
            Path to created CSV file
        """
        output_path = self.output_directory / filename
        scoped_df.to_csv(output_path, index=False)
        return output_path

    def export_instructions(
        self,
        instruction_df: pd.DataFrame,
        filename: str = "group_audit_instructions.csv"
    ) -> Path:
        """Export audit instructions to CSV.

        Args:
            instruction_df: DataFrame with audit instructions
            filename: Output filename

        Returns:
            Path to created CSV file
        """
        output_path = self.output_directory / filename
        instruction_df.to_csv(output_path, index=False)
        return output_path

    def export_coverage_summary(
        self,
        coverage_results: dict,
        filename: str = "coverage_summary.csv"
    ) -> Path:
        """Export coverage summary to CSV.

        Args:
            coverage_results: Dict with coverage metrics from CoverageCalculator
            filename: Output filename

        Returns:
            Path to created CSV file
        """
        coverage_summary = pd.DataFrame([{
            "Total_Assets": coverage_results["Total_Assets"],
            "Covered_Assets": coverage_results["Covered_Assets"],
            "Coverage_Percentage": coverage_results["Coverage_Percentage"],
            "Status": coverage_results["Status"]
        }])

        output_path = self.output_directory / filename
        coverage_summary.to_csv(output_path, index=False)
        return output_path
