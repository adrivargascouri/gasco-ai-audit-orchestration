"""BDO-style guardrails applied after ML scope prediction."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from modular.agents.risk_discovery_agent import FINANCIAL_GUARDRAIL_RISK_TYPES
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
        identified_risks: pd.DataFrame | None = None,
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

        adjusted_df = self._add_financial_risk_context(
            adjusted_df,
            identified_risks,
        )

        adjusted_rows = [
            self._apply_row_guardrails(row)
            for _, row in adjusted_df.iterrows()
        ]
        adjusted_df = pd.DataFrame(adjusted_rows)
        coverage_status = self._coverage_status(adjusted_df)
        adjusted_df = self._flag_coverage_gap(adjusted_df, coverage_status)
        return adjusted_df, coverage_status

    def _add_financial_risk_context(
        self,
        scoped_df: pd.DataFrame,
        identified_risks: pd.DataFrame | None,
    ) -> pd.DataFrame:
        result = scoped_df.copy()
        result["Financial_Risk_Types"] = ""
        result["Financial_Risk_Count"] = 0
        result["Financial_Risk_Guardrail_Applied"] = False
        result["Financial_Risk_Human_Review_Required"] = False
        result["Financial_Risk_Guardrail_Action"] = "None"
        result["Financial_Risk_Guardrail_Reason"] = ""

        if identified_risks is None or identified_risks.empty:
            return result
        required_columns = {"entity_name", "risk_type"}
        if not required_columns.issubset(identified_risks.columns):
            return result

        risk_df = identified_risks.copy()
        risk_df["_entity_key"] = self._entity_key(risk_df["entity_name"])
        risk_df["_risk_type_key"] = risk_df["risk_type"].astype(str).str.strip().str.casefold()
        canonical_by_key = {
            risk_type.casefold(): risk_type for risk_type in FINANCIAL_GUARDRAIL_RISK_TYPES
        }
        risk_df["Financial_Risk_Type"] = risk_df["_risk_type_key"].map(canonical_by_key)
        risk_df = risk_df.dropna(subset=["Financial_Risk_Type"])
        if risk_df.empty:
            return result

        summaries = []
        for entity_key, group in risk_df.groupby("_entity_key", dropna=True):
            risk_types = [
                risk_type
                for risk_type in FINANCIAL_GUARDRAIL_RISK_TYPES
                if risk_type in set(group["Financial_Risk_Type"])
            ]
            summaries.append({
                "_entity_key": entity_key,
                "Financial_Risk_Types": "; ".join(risk_types),
                "Financial_Risk_Count": int(len(group)),
            })

        summary_df = pd.DataFrame(summaries)
        result["_entity_key"] = self._entity_key(result["Entity"])
        result = result.merge(summary_df, on="_entity_key", how="left", suffixes=("", "_risk"))
        result["Financial_Risk_Types"] = result["Financial_Risk_Types_risk"].fillna(
            result["Financial_Risk_Types"]
        )
        result["Financial_Risk_Count"] = (
            pd.to_numeric(result["Financial_Risk_Count_risk"], errors="coerce")
            .fillna(result["Financial_Risk_Count"])
            .astype(int)
        )
        return result.drop(
            columns=[
                column
                for column in [
                    "_entity_key",
                    "Financial_Risk_Types_risk",
                    "Financial_Risk_Count_risk",
                ]
                if column in result
            ]
        )

    def _apply_row_guardrails(self, row: pd.Series) -> dict:
        original_scope = row["Recommended_Scope"]
        adjusted_scope = original_scope
        reasons: list[str] = []
        financial_reasons: list[str] = []
        adjusted = False
        review = False
        financial_scope_adjusted = False

        asset_percentage = float(row.get("Asset_Percentage", 0.0))
        risk_level = str(row.get("Risk_Level", ""))
        confidence = float(row.get("Prediction_Confidence", 0.0))
        severe_findings_count = int(row.get("Severe_Findings_Count", 0))
        financial_risk_types = self._financial_risk_types(row)
        financial_risk_count = int(row.get("Financial_Risk_Count", 0))

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

        requires_specific_procedures = [
            risk_type
            for risk_type in ("Liquidity Risk", "High Debt Risk")
            if risk_type in financial_risk_types
        ]
        if requires_specific_procedures:
            risk_label = " and ".join(requires_specific_procedures)
            if self._is_less_than(adjusted_scope, "Specific Procedures"):
                adjusted_scope = "Specific Procedures"
                adjusted = True
                financial_scope_adjusted = True
                financial_reasons.append(
                    f"Financial risk guardrail: {risk_label} requires at least Specific Procedures."
                )
            else:
                financial_reasons.append(
                    f"Financial risk guardrail: {risk_label} requires at least Specific Procedures; current scope already meets the minimum."
                )

        if "Manual Risk Flag" in financial_risk_types:
            review = True
            financial_reasons.append(
                "Financial risk guardrail: Manual Risk Flag requires human review."
            )

        if financial_risk_count > 1:
            review = True
            financial_reasons.append(
                "Financial risk guardrail: multiple financial risks on the same component require human review."
            )

        if confidence < self.config.confidence_threshold:
            review = True
            reasons.append(
                f"ML confidence {confidence:.2f} is below the {self.config.confidence_threshold:.2f} threshold."
            )

        if financial_reasons:
            reasons.extend(financial_reasons)

        if not reasons:
            reasons.append("ML recommendation accepted; no guardrail adjustment required.")

        action = self._action(adjusted, review, bool(financial_reasons))
        result = row.to_dict()
        result.update({
            "Original_ML_Scope": original_scope,
            "Guardrail_Adjusted_Scope": adjusted_scope,
            "Guardrail_Action": action,
            "Guardrail_Reason": " ".join(reasons),
            "Requires_Human_Review": bool(review),
            "Recommended_Scope": adjusted_scope,
            "Financial_Risk_Guardrail_Applied": bool(financial_reasons),
            "Financial_Risk_Human_Review_Required": bool(
                "Manual Risk Flag" in financial_risk_types or financial_risk_count > 1
            ),
            "Financial_Risk_Guardrail_Action": self._financial_risk_action(
                financial_scope_adjusted,
                bool(
                    "Manual Risk Flag" in financial_risk_types
                    or financial_risk_count > 1
                ),
                bool(financial_reasons),
            ),
            "Financial_Risk_Guardrail_Reason": " ".join(financial_reasons),
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

    def _entity_key(self, values: pd.Series) -> pd.Series:
        return values.astype(str).str.strip().str.casefold()

    def _financial_risk_types(self, row: pd.Series) -> set[str]:
        risk_types = str(row.get("Financial_Risk_Types", "")).strip()
        if not risk_types:
            return set()
        return {risk_type.strip() for risk_type in risk_types.split(";") if risk_type.strip()}

    def _financial_risk_action(
        self,
        adjusted: bool,
        review: bool,
        applied: bool,
    ) -> str:
        if not applied:
            return "None"
        if adjusted and review:
            return "Scope adjusted and human review required"
        if adjusted:
            return "Scope adjusted"
        if review:
            return "Human review required"
        return "Documented minimum-scope guardrail"

    def _action(self, adjusted: bool, review: bool, triggered: bool = False) -> str:
        if adjusted and review:
            return "Adjusted and flagged for review"
        if adjusted:
            return "Adjusted"
        if review:
            return "Flagged for review"
        if triggered:
            return "Guardrail applied"
        return "Accepted"
