"""Local inference for the trained GASCO scope model."""
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import pandas as pd

from modular.features.encoders import TargetScopeEncoder
from modular.model.base import ModelInference
from modular.model.model_registry import (
    SCOPE_FEATURE_COLUMNS,
    SCOPE_MODEL_PATH,
    load_model_artifact,
)

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)


class ScopeModelInference(ModelInference[Any, dict[str, Any]]):
    """Run local predictions with the saved scope model."""

    def __init__(self, model_path: Path = SCOPE_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = load_model_artifact(self.model_path)
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        self.feature_columns = list(
            getattr(self.model, "feature_columns_", SCOPE_FEATURE_COLUMNS)
        )

    @property
    def name(self) -> str:
        """Model name for identification."""
        return "gasco_scope_random_forest"

    @property
    def version(self) -> str:
        """Model version."""
        return "phase_2b_local"

    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        return self.model is not None

    def infer(self, input_data: Any) -> dict[str, Any]:
        """Predict a scope class and confidence for one input."""
        frame = self._to_frame(input_data)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`sklearn.utils.parallel.delayed`.*",
                category=UserWarning,
            )
            predicted_class = int(self.model.predict(frame)[0])
            confidence = self._confidence(frame)

        return {
            "predicted_class": predicted_class,
            "predicted_label": TargetScopeEncoder.decode(predicted_class),
            "confidence": confidence,
        }

    def batch_infer(self, batch_data: list[Any]) -> list[dict[str, Any]]:
        """Run inference on a batch of inputs."""
        return [self.infer(item) for item in batch_data]

    def _to_frame(self, input_data: Any) -> pd.DataFrame:
        if isinstance(input_data, pd.DataFrame):
            missing_columns = set(self.feature_columns) - set(input_data.columns)
            if missing_columns:
                raise ValueError(f"Missing feature columns: {sorted(missing_columns)}")
            return input_data[self.feature_columns]

        if isinstance(input_data, Mapping):
            missing_columns = set(self.feature_columns) - set(input_data.keys())
            if missing_columns:
                raise ValueError(f"Missing feature columns: {sorted(missing_columns)}")
            return pd.DataFrame([input_data], columns=self.feature_columns)

        if isinstance(input_data, Sequence) and not isinstance(input_data, (str, bytes)):
            values = list(input_data)
            if len(values) != len(self.feature_columns):
                raise ValueError(
                    f"Expected {len(self.feature_columns)} feature values, got {len(values)}"
                )
            return pd.DataFrame([values], columns=self.feature_columns)

        raise TypeError("Input must be a mapping, sequence, or pandas DataFrame")

    def _confidence(self, frame: pd.DataFrame) -> float:
        if not hasattr(self.model, "predict_proba"):
            return 1.0
        probabilities = self.model.predict_proba(frame)[0]
        return float(max(probabilities))


def predict_scope(
    input_data: Mapping[str, Any] | Sequence[Any] | pd.DataFrame,
    model_path: Path = SCOPE_MODEL_PATH,
) -> dict[str, Any]:
    """Convenience function for one local scope prediction."""
    return ScopeModelInference(model_path).infer(input_data)


if __name__ == "__main__":
    sample_input = {
        "revenue": 8500000,
        "assets": 35000000,
        "revenue_percentage": 17.8,
        "assets_percentage": 29.5,
        "risk_level_encoded": 3,
        "country_risk_score": 2,
        "prior_findings_count": 3,
        "severe_findings_count": 2,
        "growth_rate": 5.2,
        "liquidity_ratio": 1.45,
        "manual_risk_flag_encoded": 1,
    }
    print(predict_scope(sample_input))
