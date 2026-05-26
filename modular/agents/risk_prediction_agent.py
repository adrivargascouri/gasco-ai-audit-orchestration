"""Risk prediction agent for CrewAI-orchestrated ML scope analysis."""
from dataclasses import dataclass
from typing import Any

import pandas as pd

from modular.tools.ml_scope_prediction_tool import MLScopePredictionTool


@dataclass
class RiskPredictionResult:
    """Risk prediction stage result."""

    scoped_df: pd.DataFrame
    feature_df: pd.DataFrame
    explanations: dict[str, str]
    summary: str


class RiskPredictionAgent:
    """Analyze entities and collect ML scope predictions and explanations."""

    def __init__(self, prediction_tool: MLScopePredictionTool):
        self.prediction_tool = prediction_tool

    def analyze_entities(self, group_df: pd.DataFrame) -> RiskPredictionResult:
        """Call the ML scope tool and summarize predictions."""
        prediction_result = self.prediction_tool.predict_group(group_df)
        scoped_df = prediction_result["predictions"]
        feature_df = prediction_result["features"]
        explanations = prediction_result["explanations"]

        distribution = scoped_df["Recommended_Scope"].value_counts().to_dict()
        summary = (
            f"Generated ML scope predictions for {len(scoped_df)} entities. "
            f"Distribution: {distribution}."
        )

        return RiskPredictionResult(
            scoped_df=scoped_df,
            feature_df=feature_df,
            explanations=explanations,
            summary=summary,
        )
