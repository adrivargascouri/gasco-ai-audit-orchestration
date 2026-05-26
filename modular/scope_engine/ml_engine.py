"""ML-based scope recommendation engine."""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modular.features.encoders import ManualRiskFlagEncoder, RiskLevelEncoder
from modular.model.local_inference import ScopeModelInference
from modular.model.model_registry import (
    SCOPE_FEATURE_COLUMNS,
    SCOPE_MODEL_PATH,
    TRAINING_DATA_PATH,
    TARGET_COLUMN,
)
from modular.scope_engine.base import ScopeRecommendationEngine
from modular.domain.models import RiskLevel, ScopeRecommendation, ScopeType


ESTIMATION_FEATURE_WEIGHTS = {
    "revenue_percentage": 2.0,
    "assets_percentage": 3.0,
    "risk_level_encoded": 1.25,
    "country_risk_score": 0.5,
    "prior_findings_count": 1.0,
    "severe_findings_count": 1.0,
}


class MLScopeEngine(ScopeRecommendationEngine):
    """Scope recommendation engine backed by the local trained ML model."""

    def __init__(
        self,
        model_path: str | Path = SCOPE_MODEL_PATH,
        training_data_path: str | Path = TRAINING_DATA_PATH,
        historical_data_path: str | Path = "data/raw/historical_audit_data.csv",
        findings_data_path: str | Path = "data/findings_repo.csv",
    ):
        self.model_path = Path(model_path)
        self.training_data_path = Path(training_data_path)
        self.historical_data_path = Path(historical_data_path)
        self.findings_data_path = Path(findings_data_path)
        self.inference = ScopeModelInference(self.model_path)
        self.feature_columns = list(getattr(self.inference, "feature_columns", SCOPE_FEATURE_COLUMNS))
        if self.feature_columns != SCOPE_FEATURE_COLUMNS:
            raise ValueError(
                "Loaded model feature order does not match the training registry columns"
            )
        self.training_df = self._load_training_data()
        self.historical_df = self._load_historical_data()
        self.findings_df = self._load_findings_data()
        self.training_total_revenue = self._median_implied_total(
            "revenue",
            "revenue_percentage",
        )
        self.training_total_assets = self._median_implied_total(
            "assets",
            "assets_percentage",
        )
        self.last_feature_frame = pd.DataFrame(columns=["Entity", *SCOPE_FEATURE_COLUMNS])

    @property
    def name(self) -> str:
        """Engine name."""
        return "MLScopeEngine"

    @property
    def version(self) -> str:
        """Engine version."""
        return "2.0-local"

    def recommend_scope(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """Generate ML scope recommendations for all entities."""
        result_rows = []
        feature_debug_rows = []
        total_revenue = group_df["Revenue"].sum()
        total_assets = group_df["Assets"].sum()

        for _, row in group_df.iterrows():
            features = self._build_feature_row(row, total_revenue, total_assets)
            prediction = self.inference.infer(features)
            feature_debug_rows.append({"Entity": row["Entity"], **features})

            result_rows.append({
                "Entity": row["Entity"],
                "Country": row["Country"],
                "Assets": row["Assets"],
                "Risk_Level": row["Risk_Level"],
                "Asset_Percentage": features["assets_percentage"],
                "Recommended_Scope": prediction["predicted_label"],
                "Prediction_Confidence": round(prediction["confidence"], 4),
            })

        self.last_feature_frame = pd.DataFrame(feature_debug_rows)
        return pd.DataFrame(result_rows)

    def get_recommendation(self, entity_data: dict) -> ScopeRecommendation:
        """Get an ML scope recommendation for a single entity."""
        total_revenue = entity_data.get("total_revenue", entity_data.get("revenue", 0))
        total_assets = entity_data.get("total_assets", entity_data.get("assets", 0))
        row = pd.Series({
            "Entity": entity_data["entity"],
            "Country": entity_data["country"],
            "Revenue": entity_data["revenue"],
            "Assets": entity_data["assets"],
            "Risk_Level": entity_data["risk_level"],
        })
        features = self._build_feature_row(row, total_revenue, total_assets)
        prediction = self.inference.infer(features)

        return ScopeRecommendation(
            entity=entity_data["entity"],
            country=entity_data["country"],
            assets=entity_data["assets"],
            asset_percentage=features["assets_percentage"],
            risk_level=RiskLevel(entity_data["risk_level"]),
            recommended_scope=ScopeType(prediction["predicted_label"]),
        )

    def _build_feature_row(
        self,
        row: pd.Series,
        total_revenue: float,
        total_assets: float,
    ) -> dict[str, Any]:
        risk_level = row["Risk_Level"]
        entity = row["Entity"]
        severe_findings_count = self._severe_findings_count(entity)
        prior_findings_count = self._prior_findings_count(entity)
        revenue_percentage = self._percentage(row["Revenue"], total_revenue)
        assets_percentage = self._percentage(row["Assets"], total_assets)
        risk_level_encoded = RiskLevelEncoder.encode(risk_level)
        country_risk_score = self._country_risk_score(row["Country"])
        available_features = {
            "revenue_percentage": revenue_percentage,
            "assets_percentage": assets_percentage,
            "risk_level_encoded": risk_level_encoded,
            "country_risk_score": country_risk_score,
            "prior_findings_count": prior_findings_count,
            "severe_findings_count": severe_findings_count,
        }
        estimates = self._estimate_missing_features(available_features)
        manual_risk_flag_encoded = estimates["manual_risk_flag_encoded"]
        if risk_level == RiskLevel.HIGH.value or severe_findings_count > 0:
            manual_risk_flag_encoded = 1

        features = {
            "revenue": self._model_scale_amount(
                revenue_percentage,
                self.training_total_revenue,
            ),
            "assets": self._model_scale_amount(
                assets_percentage,
                self.training_total_assets,
            ),
            "revenue_percentage": revenue_percentage,
            "assets_percentage": assets_percentage,
            "risk_level_encoded": risk_level_encoded,
            "country_risk_score": country_risk_score,
            "prior_findings_count": prior_findings_count,
            "severe_findings_count": severe_findings_count,
            "growth_rate": estimates["growth_rate"],
            "liquidity_ratio": estimates["liquidity_ratio"],
            "manual_risk_flag_encoded": manual_risk_flag_encoded,
        }

        return {column: features[column] for column in SCOPE_FEATURE_COLUMNS}

    def _load_training_data(self) -> pd.DataFrame:
        if not self.training_data_path.exists():
            raise FileNotFoundError(f"Training dataset not found: {self.training_data_path}")

        training_df = pd.read_csv(self.training_data_path)
        required_columns = set(SCOPE_FEATURE_COLUMNS + [TARGET_COLUMN])
        missing_columns = required_columns - set(training_df.columns)
        if missing_columns:
            raise ValueError(f"Training dataset missing columns: {sorted(missing_columns)}")
        return training_df

    def _load_historical_data(self) -> pd.DataFrame:
        if not self.historical_data_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.historical_data_path)

    def _load_findings_data(self) -> pd.DataFrame:
        if not self.findings_data_path.exists():
            return pd.DataFrame(columns=["Entity", "Severity"])
        return pd.read_csv(self.findings_data_path)

    def _percentage(self, value: float, total: float) -> float:
        return round((value / total * 100), 4) if total else 0.0

    def _median_implied_total(self, amount_column: str, percentage_column: str) -> float:
        valid_rows = self.training_df[self.training_df[percentage_column] > 0]
        implied_totals = valid_rows[amount_column] / (valid_rows[percentage_column] / 100)
        return float(implied_totals.median())

    def _model_scale_amount(self, percentage: float, training_total: float) -> int:
        return int(round(training_total * (percentage / 100)))

    def _country_risk_score(self, country: str) -> int:
        if self.historical_df.empty:
            return 3

        country_aliases = {"USA": "US"}
        lookup_country = country_aliases.get(country, country)
        matches = self.historical_df[
            self.historical_df["country"].str.lower() == lookup_country.lower()
        ]
        if not matches.empty:
            return int(matches["country_risk_score"].median())
        return int(self.historical_df["country_risk_score"].median())

    def _estimate_missing_features(self, available_features: dict[str, Any]) -> dict[str, Any]:
        neighbors = self._nearest_training_neighbors(available_features)
        manual_risk_probability = float(neighbors["manual_risk_flag_encoded"].mean())
        return {
            "growth_rate": round(float(neighbors["growth_rate"].median()), 1),
            "liquidity_ratio": round(float(neighbors["liquidity_ratio"].median()), 2),
            "manual_risk_flag_encoded": ManualRiskFlagEncoder.encode(
                "Yes" if manual_risk_probability >= 0.5 else "No"
            ),
        }

    def _nearest_training_neighbors(
        self,
        available_features: dict[str, Any],
        neighbor_count: int = 7,
    ) -> pd.DataFrame:
        columns = list(ESTIMATION_FEATURE_WEIGHTS)
        training_slice = self.training_df[columns]
        ranges = (training_slice.max() - training_slice.min()).replace(0, 1)
        query = pd.Series(available_features, index=columns, dtype=float)
        weights = pd.Series(ESTIMATION_FEATURE_WEIGHTS, index=columns, dtype=float)

        distances = (((training_slice - query) / ranges) * weights).pow(2).sum(axis=1)
        distances = np.sqrt(distances)
        neighbor_indexes = distances.nsmallest(neighbor_count).index
        return self.training_df.loc[neighbor_indexes]

    def _prior_findings_count(self, entity: str) -> int:
        if self.findings_df.empty:
            return 0
        return int((self.findings_df["Entity"] == entity).sum())

    def _severe_findings_count(self, entity: str) -> int:
        if self.findings_df.empty:
            return 0
        entity_findings = self.findings_df[self.findings_df["Entity"] == entity]
        return int((entity_findings["Severity"] == RiskLevel.HIGH.value).sum())
