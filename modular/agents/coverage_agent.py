"""Coverage agent for CrewAI-orchestrated audit evaluation."""
from dataclasses import dataclass

import pandas as pd

from modular.config.config_schema import CoverageConfig


@dataclass
class CoverageResult:
    """Coverage stage result."""

    summary: dict
    risky_uncovered_df: pd.DataFrame
    narrative: str


class CoverageAgent:
    """Evaluate audit coverage and identify risky uncovered entities."""

    def __init__(self, config: CoverageConfig):
        self.config = config

    def evaluate_coverage(self, scoped_df: pd.DataFrame) -> CoverageResult:
        """Calculate covered assets and risky uncovered entities."""
        scope_column = (
            "Guardrail_Adjusted_Scope"
            if "Guardrail_Adjusted_Scope" in scoped_df.columns
            else "Recommended_Scope"
        )
        total_assets = scoped_df["Assets"].sum()
        covered_df = scoped_df[
            scoped_df[scope_column].isin(self.config.included_scopes)
        ]
        covered_assets = covered_df["Assets"].sum()
        coverage_percentage = (
            covered_assets / total_assets * 100 if total_assets else 0.0
        )
        status = (
            "Sufficient audit coverage"
            if coverage_percentage >= self.config.minimum_coverage_percentage
            else "Warning: Coverage is too low"
        )

        uncovered_df = scoped_df[
            ~scoped_df[scope_column].isin(self.config.included_scopes)
        ].copy()
        risky_uncovered_df = uncovered_df[
            (uncovered_df["Risk_Level"] == "High")
            | (uncovered_df["Asset_Percentage"] >= 5.0)
        ].copy()

        summary = {
            "Total_Assets": total_assets,
            "Covered_Assets": covered_assets,
            "Coverage_Percentage": coverage_percentage,
            "Status": status,
        }
        narrative = (
            f"Coverage is {coverage_percentage:.2f}% with status: {status}. "
            f"Risky uncovered entities identified: {len(risky_uncovered_df)}."
        )

        return CoverageResult(
            summary=summary,
            risky_uncovered_df=risky_uncovered_df,
            narrative=narrative,
        )
