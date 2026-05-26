"""Coverage calculation engine."""
import pandas as pd
from ..config.config_schema import CoverageConfig
from ..scope_engine.base import ScopeRecommendationEngine


class CoverageCalculator:
    """Calculates audit coverage for the group.

    Determines what percentage of group assets are covered by Full Scope
    or Specific Procedures audits, based on configurable scope types.
    """

    def __init__(self, config: CoverageConfig, scope_engine: ScopeRecommendationEngine):
        """Initialize coverage calculator.

        Args:
            config: CoverageConfig with minimum_coverage_percentage and included_scopes
            scope_engine: ScopeRecommendationEngine to generate scope recommendations
        """
        self.config = config
        self.scope_engine = scope_engine

    def calculate_coverage(self, group_df: pd.DataFrame) -> dict:
        """Calculate audit coverage for the group.

        Args:
            group_df: DataFrame with entity data

        Returns:
            Dict with Total_Assets, Covered_Assets, Coverage_Percentage, Status
        """
        # Get scope recommendations
        scoped_df = self.scope_engine.recommend_scope(group_df)

        # Total group assets
        total_assets = scoped_df["Assets"].sum()

        # Components included in audit coverage
        covered_components = scoped_df[
            scoped_df["Recommended_Scope"].isin(self.config.included_scopes)
        ]

        # Covered assets
        covered_assets = covered_components["Assets"].sum()

        # Coverage percentage
        coverage_percentage = (covered_assets / total_assets * 100) if total_assets > 0 else 0

        # Determine status
        status = (
            "Sufficient audit coverage"
            if coverage_percentage >= self.config.minimum_coverage_percentage
            else "Warning: Coverage is too low"
        )

        return {
            "Total_Assets": total_assets,
            "Covered_Assets": covered_assets,
            "Coverage_Percentage": coverage_percentage,
            "Status": status,
            "Covered_Components": covered_components
        }
