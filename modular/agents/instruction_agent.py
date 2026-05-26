"""Instruction agent for CrewAI-orchestrated audit narratives."""
from dataclasses import dataclass

import pandas as pd

from modular.config.config_schema import AuditConfig
from modular.domain.models import ScopeType


@dataclass
class InstructionResult:
    """Instruction stage result."""

    instruction_df: pd.DataFrame
    final_narrative: str


class InstructionAgent:
    """Generate audit instructions and summarize ML explanations."""

    def __init__(self, config: AuditConfig):
        self.config = config

    def generate_instructions(
        self,
        scoped_df: pd.DataFrame,
        explanations: dict[str, str],
        coverage_summary: dict,
    ) -> InstructionResult:
        """Build audit instruction rows from ML scope recommendations."""
        rows = []
        for _, row in scoped_df.iterrows():
            entity = row["Entity"]
            scope = row["Recommended_Scope"]
            rows.append({
                "Entity": entity,
                "Country": row["Country"],
                "Risk_Level": row["Risk_Level"],
                "Recommended_Scope": scope,
                "Prediction_Confidence": row["Prediction_Confidence"],
                "Audit_Instruction": self._instruction(entity, row["Country"], scope),
                "ML_Explanation_Summary": self._short_explanation(
                    explanations.get(entity, "")
                ),
            })

        instruction_df = pd.DataFrame(rows)
        final_narrative = (
            "CrewAI-orchestrated GASCO ML audit workflow completed. "
            f"Coverage status: {coverage_summary['Status']} "
            f"({coverage_summary['Coverage_Percentage']:.2f}%). "
            f"Prepared {len(instruction_df)} component audit instructions with ML explanations."
        )

        return InstructionResult(instruction_df=instruction_df, final_narrative=final_narrative)

    def _instruction(self, entity: str, country: str, scope: str) -> str:
        template_map = {
            ScopeType.FULL_SCOPE.value: self.config.full_scope_template,
            ScopeType.SPECIFIC_PROCEDURES.value: self.config.specific_procedures_template,
            ScopeType.ANALYTICAL_PROCEDURES.value: self.config.analytical_procedures_template,
        }
        template = template_map.get(scope, self.config.analytical_procedures_template)
        return template.format(entity=entity, country=country)

    def _short_explanation(self, explanation: str) -> str:
        lines = [line for line in explanation.splitlines() if line.startswith("- ")]
        return " ".join(line[2:] for line in lines[:3])
