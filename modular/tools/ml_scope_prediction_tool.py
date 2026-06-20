"""CrewAI-compatible tool wrapping MLScopeEngine predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from modular.model.explainability import ScopeModelExplainability
from modular.scope_engine.ml_engine import MLScopeEngine


class MLScopePredictionInput(BaseModel):
    """Input schema for CrewAI tool calls."""

    entities_json: str = Field(
        ...,
        description="JSON array of group entities with Entity, Country, Revenue, Assets, Risk_Level.",
    )


class MLScopePredictionTool(BaseTool):
    """Predict audit scopes with the trained local ML model."""

    name: str = "ml_scope_prediction_tool"
    description: str = (
        "Predict audit scope for GASCO group entities using the local "
        "MLScopeEngine and return predicted scope, confidence, and explanation."
    )
    args_schema: type[BaseModel] = MLScopePredictionInput

    def __init__(
        self,
        findings_data_path: str | Path = "data/findings_repo.csv",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        object.__setattr__(
            self,
            "scope_engine",
            MLScopeEngine(findings_data_path=findings_data_path),
        )
        object.__setattr__(self, "explainer", ScopeModelExplainability())

    def _run(self, entities_json: str) -> str:
        entities = json.loads(entities_json)
        group_df = pd.DataFrame(entities)
        result = self.predict_group(group_df)
        return result["predictions"].to_json(orient="records")

    def predict_group(self, group_df: pd.DataFrame) -> dict[str, Any]:
        """Predict scopes and explanations for all group entities."""
        scoped_df = self.scope_engine.recommend_scope(group_df)
        feature_df = self.scope_engine.last_feature_frame.copy()

        explanations = {}
        for _, feature_row in feature_df.iterrows():
            entity = feature_row["Entity"]
            explanations[entity] = self.explainer.explain_prediction(entity, feature_row)

        predictions_df = scoped_df.copy()
        predictions_df["Explanation"] = predictions_df["Entity"].map(explanations)

        return {
            "predictions": predictions_df,
            "features": feature_df,
            "explanations": explanations,
        }

    def predict_entity(self, entity_data: dict[str, Any], group_df: pd.DataFrame) -> dict[str, Any]:
        """Predict scope and explanation for a single entity within a group."""
        result = self.predict_group(group_df)
        predictions = result["predictions"]
        entity_row = predictions[predictions["Entity"] == entity_data["Entity"]]
        if entity_row.empty:
            raise ValueError(f"Entity not found in prediction output: {entity_data['Entity']}")

        row = entity_row.iloc[0]
        return {
            "predicted_scope": row["Recommended_Scope"],
            "confidence": float(row["Prediction_Confidence"]),
            "explanation": result["explanations"][entity_data["Entity"]],
        }
