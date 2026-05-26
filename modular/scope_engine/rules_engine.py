"""Rules-based scope recommendation engine."""
import pandas as pd
from ..scope_engine.base import ScopeRecommendationEngine
from ..config.config_schema import ScopeRulesConfig
from ..domain.models import ScopeRecommendation, RiskLevel, ScopeType


class RulesScopeEngine(ScopeRecommendationEngine):
    """Rules-based engine for scope recommendations.

    Recommends audit scope based on asset percentage and risk level,
    using configurable thresholds from config.
    """

    def __init__(self, config: ScopeRulesConfig):
        """Initialize rules engine with configuration.

        Args:
            config: ScopeRulesConfig with thresholds and labels
        """
        self.config = config

    @property
    def name(self) -> str:
        """Engine name."""
        return "RulesScopeEngine"

    @property
    def version(self) -> str:
        """Engine version."""
        return "1.0"

    def recommend_scope(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """Generate scope recommendations for all entities.

        Args:
            group_df: DataFrame with columns [Entity, Country, Assets, Risk_Level, ...]

        Returns:
            DataFrame with added Asset_Percentage and Recommended_Scope columns
        """
        result_df = group_df.copy()
        total_assets = result_df["Assets"].sum()

        # Add asset percentage column
        result_df["Asset_Percentage"] = (result_df["Assets"] / total_assets) * 100

        # Apply scope classification rules
        result_df["Recommended_Scope"] = result_df.apply(
            lambda row: self._classify_component(row, total_assets),
            axis=1
        )

        return result_df

    def _classify_component(self, row: pd.Series, total_assets: float) -> str:
        """Classify a single component's scope based on rules.

        Rules (in order):
        1. If asset_percentage > threshold -> Full Scope
        2. Else if risk_level == High -> Specific Procedures
        3. Else -> Analytical Procedures

        Args:
            row: DataFrame row with Entity data
            total_assets: Total group assets for percentage calculation

        Returns:
            Scope recommendation string
        """
        asset_percentage = row["Assets"] / total_assets

        if asset_percentage > self.config.asset_percentage_threshold:
            return self.config.full_scope_label
        elif row["Risk_Level"] == RiskLevel.HIGH.value:
            return self.config.high_risk_scope
        else:
            return self.config.default_scope

    def get_recommendation(self, entity_data: dict) -> ScopeRecommendation:
        """Get scope recommendation for a single entity.

        Args:
            entity_data: Dict with entity, country, assets, risk_level, total_assets

        Returns:
            ScopeRecommendation object
        """
        total_assets = entity_data.get("total_assets", 1)
        asset_percentage = (entity_data["assets"] / total_assets) * 100

        if asset_percentage / 100 > self.config.asset_percentage_threshold:
            scope = ScopeType.FULL_SCOPE
        elif entity_data["risk_level"] == RiskLevel.HIGH.value:
            scope = ScopeType.SPECIFIC_PROCEDURES
        else:
            scope = ScopeType.ANALYTICAL_PROCEDURES

        return ScopeRecommendation(
            entity=entity_data["entity"],
            country=entity_data["country"],
            assets=entity_data["assets"],
            asset_percentage=asset_percentage,
            risk_level=RiskLevel(entity_data["risk_level"]),
            recommended_scope=scope
        )
