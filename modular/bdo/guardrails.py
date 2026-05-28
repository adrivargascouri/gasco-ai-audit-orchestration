"""BDO-style guardrails applied after ML scope prediction."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from modular.config.config_schema import BDOGuardrailsConfig


@dataclass
class GuardrailCoverageStatus:
    """Coverage status after guardrail-adjusted scope."""

    total_assets: float
    covered_assets: float
    coverage_percentage: float
    status: str


class BDOGuardrails:
    """Validate and adjust ML recommendations using methodology guardrails."""

    def __init__(self, config: BDOGuardrailsConfig):
        self.config = config

    def apply(
        self,
        scoped_df: pd.DataFrame,
        feature_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, GuardrailCoverageStatus]:
        """Apply guardrails and return adjusted recommendations plus coverage."""
        adjusted_df = scoped_df.copy()
        if feature_df is not None and "severe_findings_count" in feature_df.columns:
            severe_counts = feature_df.set_index("Entity")["severe_findings_count"]
            adjusted_df["Severe_Findings_Count"] = (
                adjusted_df["Entity"].map(severe_counts).fillna(0).astype(int)
            )
        elif "Severe_Findings_Count" not in adjusted_df.columns:
            adjusted_df["Severe_Findings_Count"] = 0

        adjusted_rows = [
            self._apply_row_guardrails(row)
            for _, row in adjusted_df.iterrows()
        ]
        adjusted_df = pd.DataFrame(adjusted_rows)
        coverage_status = self._coverage_status(adjusted_df)
        adjusted_df = self._flag_coverage_gap(adjusted_df, coverage_status)
        return adjusted_df, coverage_status

    def _apply_row_guardrails(self, row: pd.Series) -> dict:
        original_scope = row["Recommended_Scope"]
        adjusted_scope = original_scope
        reasons: list[str] = []
        adjusted = False
        review = False

        asset_percentage = float(row.get("Asset_Percentage", 0.0))
        risk_level = str(row.get("Risk_Level", ""))
        confidence = float(row.get("Prediction_Confidence", 0.0))
        severe_findings_count = int(row.get("Severe_Findings_Count", 0))

        if asset_percentage >= self.config.significant_component_threshold:
            if self._is_less_than(adjusted_scope, "Full Scope"):
                adjusted_scope = "Full Scope"
                adjusted = True
                review = True
                reasons.append(
                    "Significant component threshold met; scope increased to Full Scope for partner review."
                )

        if risk_level in self.config.high_risk_levels:
            if self._is_less_than(adjusted_scope, "Specific Procedures"):
                adjusted_scope = "Specific Procedures"
                adjusted = True
                reasons.append(
                    "High or critical risk level requires at least Specific Procedures."
                )

        if severe_findings_count > 0:
            if adjusted_scope == "Analytical Procedures":
                adjusted_scope = "Specific Procedures"
                adjusted = True
                review = True
                reasons.append(
                    "Severe prior findings prevent analytical-only coverage without review."
                )
            else:
                reasons.append(
                    "Severe prior findings noted; planned work should address the finding history."
                )

        if confidence < self.config.confidence_threshold:
            review = True
            reasons.append(
                f"ML confidence {confidence:.2f} is below the {self.config.confidence_threshold:.2f} threshold."
            )

        if not reasons:
            reasons.append("ML recommendation accepted; no guardrail adjustment required.")

        action = self._action(adjusted, review)
        result = row.to_dict()
        result.update({
            "Original_ML_Scope": original_scope,
            "Guardrail_Adjusted_Scope": adjusted_scope,
            "Guardrail_Action": action,
            "Guardrail_Reason": " ".join(reasons),
            "Requires_Human_Review": bool(review),
            "Recommended_Scope": adjusted_scope,
        })
        return result

    def _coverage_status(self, scoped_df: pd.DataFrame) -> GuardrailCoverageStatus:
        included_scopes = {
            scope for scope, order in self.config.scope_order.items() if order >= 2
        }
        total_assets = float(scoped_df["Assets"].sum())
        covered_assets = float(
            scoped_df[
                scoped_df["Guardrail_Adjusted_Scope"].isin(included_scopes)
            ]["Assets"].sum()
        )
        coverage_percentage = (
            covered_assets / total_assets * 100 if total_assets else 0.0
        )
        status = (
            "Sufficient audit coverage"
            if coverage_percentage >= self.config.minimum_coverage_percentage
            else "Warning: Guardrail-adjusted coverage is too low"
        )
        return GuardrailCoverageStatus(
            total_assets=total_assets,
            covered_assets=covered_assets,
            coverage_percentage=coverage_percentage,
            status=status,
        )

    def _flag_coverage_gap(
        self,
        scoped_df: pd.DataFrame,
        coverage_status: GuardrailCoverageStatus,
    ) -> pd.DataFrame:
        if coverage_status.coverage_percentage >= self.config.minimum_coverage_percentage:
            return scoped_df

        result = scoped_df.copy()
        uncovered_mask = result["Guardrail_Adjusted_Scope"].map(
            lambda scope: self.config.scope_order.get(scope, 0) < 2
        )
        reason = (
            f" Group coverage {coverage_status.coverage_percentage:.2f}% is below "
            f"the {self.config.minimum_coverage_percentage:.2f}% minimum; coverage selection needs review."
        )
        result.loc[uncovered_mask, "Requires_Human_Review"] = True
        result.loc[uncovered_mask, "Guardrail_Reason"] = (
            result.loc[uncovered_mask, "Guardrail_Reason"] + reason
        )
        result.loc[
            uncovered_mask & (result["Guardrail_Action"] == "Accepted"),
            "Guardrail_Action",
        ] = "Flagged for review"
        return result

    def _is_less_than(self, current_scope: str, required_scope: str) -> bool:
        return (
            self.config.scope_order.get(current_scope, 0)
            < self.config.scope_order.get(required_scope, 0)
        )

    def _action(self, adjusted: bool, review: bool) -> str:
        if adjusted and review:
            return "Adjusted and flagged for review"
        if adjusted:
            return "Adjusted"
        if review:
            return "Flagged for review"
        return "Accepted"
