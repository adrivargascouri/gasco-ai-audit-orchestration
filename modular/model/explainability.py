"""Explainability helpers for the local GASCO scope model."""
from pathlib import Path
from typing import Any, Mapping
import warnings

import pandas as pd

from modular.features.encoders import RiskLevelEncoder
from modular.model.local_inference import ScopeModelInference
from modular.model.model_registry import (
    SCOPE_FEATURE_COLUMNS,
    SCOPE_MODEL_PATH,
    TRAINING_DATA_PATH,
)

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)


class ScopeModelExplainability:
    """Feature importance and prediction explanation utilities."""

    def __init__(
        self,
        model_path: str | Path = SCOPE_MODEL_PATH,
        training_data_path: str | Path = TRAINING_DATA_PATH,
    ):
        self.model_path = Path(model_path)
        self.training_data_path = Path(training_data_path)
        self.inference = ScopeModelInference(self.model_path)
        self.model = self.inference.model
        self.feature_columns = list(
            getattr(self.model, "feature_columns_", SCOPE_FEATURE_COLUMNS)
        )
        self.training_df = pd.read_csv(self.training_data_path)

    def feature_importance(self) -> pd.DataFrame:
        """Return RandomForest feature importances sorted descending."""
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Loaded model does not expose feature_importances_")

        importance_df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_,
        })
        return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save_feature_importance(self, output_path: str | Path) -> Path:
        """Save sorted feature importances to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_importance().to_csv(output_path, index=False)
        return output_path

    def explain_prediction(
        self,
        entity_name: str,
        feature_row: Mapping[str, Any] | pd.Series,
        top_n: int = 5,
    ) -> str:
        """Explain one entity prediction in human-readable audit language."""
        features = self._coerce_feature_dict(feature_row)
        prediction = self.inference.infer(features)
        probabilities = self._prediction_probabilities(features)
        reasons = self._top_reasons(features, top_n=top_n)

        lines = [
            (
                f"{entity_name} was predicted as {prediction['predicted_label']} "
                f"with {prediction['confidence']:.2%} confidence."
            ),
            "The model likely weighted these signals most heavily:",
        ]
        lines.extend(f"- {reason}" for reason in reasons)
        lines.append(
            "Class probabilities: "
            f"Analytical Procedures {probabilities.get('Analytical Procedures', 0.0):.2%}, "
            f"Specific Procedures {probabilities.get('Specific Procedures', 0.0):.2%}, "
            f"Full Scope {probabilities.get('Full Scope', 0.0):.2%}."
        )
        return "\n".join(lines)

    def save_prediction_explanations(
        self,
        feature_df: pd.DataFrame,
        entity_names: list[str],
        output_path: str | Path,
    ) -> Path:
        """Save prediction explanations for selected entities."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sections = []
        for entity_name in entity_names:
            entity_rows = feature_df[feature_df["Entity"] == entity_name]
            if entity_rows.empty:
                sections.append(f"{entity_name}: feature vector not found.")
                continue
            sections.append(
                self.explain_prediction(entity_name, entity_rows.iloc[0])
            )

        output_path.write_text("\n\n".join(sections), encoding="utf-8")
        return output_path

    def _coerce_feature_dict(
        self,
        feature_row: Mapping[str, Any] | pd.Series,
    ) -> dict[str, Any]:
        return {column: feature_row[column] for column in self.feature_columns}

    def _prediction_probabilities(self, features: dict[str, Any]) -> dict[str, float]:
        frame = pd.DataFrame([features], columns=self.feature_columns)
        if not hasattr(self.model, "predict_proba"):
            return {}

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`sklearn.utils.parallel.delayed`.*",
                category=UserWarning,
            )
            probabilities = self.model.predict_proba(frame)[0]
        labels = getattr(self.model, "target_labels_", {})
        return {
            labels.get(int(class_id), str(class_id)): float(probability)
            for class_id, probability in zip(self.model.classes_, probabilities)
        }

    def _top_reasons(self, features: dict[str, Any], top_n: int) -> list[str]:
        importance_df = self.feature_importance()
        reasons = []

        for feature in importance_df["feature"]:
            reason = self._describe_feature_signal(feature, features[feature])
            if reason:
                reasons.append(reason)
            if len(reasons) >= top_n:
                break

        return reasons

    def _describe_feature_signal(self, feature: str, value: Any) -> str:
        importance = self._importance_for(feature)

        if feature == "risk_level_encoded":
            risk_level = RiskLevelEncoder.decode(int(value))
            return f"{risk_level.lower()} risk level; risk_level_encoded is a top model feature ({importance:.3f})"

        if feature == "liquidity_ratio":
            q25, q75 = self._quantiles(feature)
            if value <= q25:
                return f"low liquidity_ratio ({value:.2f}), consistent with higher audit attention ({importance:.3f})"
            if value >= q75:
                return f"strong liquidity_ratio ({value:.2f}), consistent with lower audit attention ({importance:.3f})"
            return f"moderate liquidity_ratio ({value:.2f}) ({importance:.3f})"

        if feature == "manual_risk_flag_encoded":
            flag = "enabled" if int(value) == 1 else "not enabled"
            return f"manual risk flag {flag}; this is a high-importance model signal ({importance:.3f})"

        if feature == "assets_percentage":
            q25, q75 = self._quantiles(feature)
            if value >= q75:
                return f"high assets_percentage ({value:.2f}%), above the training upper quartile ({q75:.2f}%)"
            if value <= q25:
                return f"low assets_percentage ({value:.2f}%), below the training lower quartile ({q25:.2f}%)"
            return f"mid-range assets_percentage ({value:.2f}%)"

        if feature == "assets":
            q75 = self.training_df[feature].quantile(0.75)
            if value >= q75:
                return f"large model-scale assets ({value:,.0f}), above the training upper quartile"
            return f"model-scale assets ({value:,.0f})"

        if feature == "revenue_percentage":
            q25, q75 = self._quantiles(feature)
            if value >= q75:
                return f"high revenue_percentage ({value:.2f}%), above the training upper quartile ({q75:.2f}%)"
            if value <= q25:
                return f"low revenue_percentage ({value:.2f}%), below the training lower quartile ({q25:.2f}%)"
            return f"mid-range revenue_percentage ({value:.2f}%)"

        if feature == "growth_rate":
            q25, q75 = self._quantiles(feature)
            if value >= q75:
                return f"elevated growth_rate ({value:.1f}), above the training upper quartile ({q75:.1f})"
            if value <= q25:
                return f"low growth_rate ({value:.1f}), below the training lower quartile ({q25:.1f})"
            return f"moderate growth_rate ({value:.1f})"

        if feature == "prior_findings_count":
            return f"prior findings count of {int(value)}"

        if feature == "severe_findings_count":
            return f"severe findings count of {int(value)}"

        if feature == "country_risk_score":
            q75 = self.training_df[feature].quantile(0.75)
            if value >= q75:
                return f"elevated country_risk_score ({int(value)})"
            return f"country_risk_score of {int(value)}"

        if feature == "revenue":
            return f"model-scale revenue ({value:,.0f})"

        return ""

    def _importance_for(self, feature: str) -> float:
        importance_df = self.feature_importance()
        return float(importance_df.loc[importance_df["feature"] == feature, "importance"].iloc[0])

    def _quantiles(self, feature: str) -> tuple[float, float]:
        return (
            float(self.training_df[feature].quantile(0.25)),
            float(self.training_df[feature].quantile(0.75)),
        )


def get_feature_importance() -> pd.DataFrame:
    """Convenience function for sorted scope model feature importances."""
    return ScopeModelExplainability().feature_importance()
