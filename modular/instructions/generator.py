"""Audit instruction generation engine."""
import pandas as pd
from ..config.config_schema import AuditConfig
from ..scope_engine.base import ScopeRecommendationEngine
from ..domain.models import ScopeType


class InstructionGenerator:
    """Generates audit instructions based on scope recommendations.

    Uses templates from config to generate instruction text for each entity,
    formatted consistently with scope recommendations.
    """

    def __init__(self, config: AuditConfig, scope_engine: ScopeRecommendationEngine):
        """Initialize instruction generator.

        Args:
            config: AuditConfig with instruction templates
            scope_engine: ScopeRecommendationEngine to get scope recommendations
        """
        self.config = config
        self.scope_engine = scope_engine

    def generate_instructions(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """Generate audit instructions for all entities.

        Args:
            group_df: DataFrame with entity data

        Returns:
            DataFrame with Entity, Country, Risk_Level, Recommended_Scope, Audit_Instruction
        """
        # Get scope recommendations
        scoped_df = self.scope_engine.recommend_scope(group_df)

        instructions = []

        for _, row in scoped_df.iterrows():
            entity = row["Entity"]
            country = row["Country"]
            risk_level = row["Risk_Level"]
            scope = row["Recommended_Scope"]

            instruction = self._generate_instruction(entity, country, scope)

            instructions.append({
                "Entity": entity,
                "Country": country,
                "Risk_Level": risk_level,
                "Recommended_Scope": scope,
                "Audit_Instruction": instruction
            })

        return pd.DataFrame(instructions)

    def _generate_instruction(self, entity: str, country: str, scope: str) -> str:
        """Generate instruction text for a specific entity and scope.

        Args:
            entity: Entity name
            country: Country of operation
            scope: Recommended scope type

        Returns:
            Formatted instruction text
        """
        template_map = {
            ScopeType.FULL_SCOPE.value: self.config.full_scope_template,
            ScopeType.SPECIFIC_PROCEDURES.value: self.config.specific_procedures_template,
            ScopeType.ANALYTICAL_PROCEDURES.value: self.config.analytical_procedures_template,
        }

        template = template_map.get(
            scope,
            self.config.analytical_procedures_template
        )

        return template.format(entity=entity, country=country)
