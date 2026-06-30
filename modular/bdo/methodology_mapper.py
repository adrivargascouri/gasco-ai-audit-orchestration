"""Map GASCO outputs to BDO-style group audit concepts."""
from __future__ import annotations

import pandas as pd


class BDOMethodologyMapper:
    """Add methodology-facing fields to adjusted GASCO recommendations."""

    def map_outputs(self, scoped_df: pd.DataFrame) -> pd.DataFrame:
        mapped_df = scoped_df.copy()
        mapped_df["Audit_Phase"] = "Scoping"
        mapped_df["Risk_Response_Level"] = mapped_df.apply(
            self._risk_response_level,
            axis=1,
        )
        mapped_df["Component_Selected_For_Work"] = mapped_df[
            "Guardrail_Adjusted_Scope"
        ].map(lambda scope: "No" if scope == "Analytical Procedures" else "Yes")
        mapped_df["Component_Auditor_Involvement"] = mapped_df[
            "Component_Selected_For_Work"
        ].map(lambda selected: "Required" if selected == "Yes" else "Not Required")
        mapped_df["Basis_For_Selection"] = mapped_df.apply(
            self._basis_for_selection,
            axis=1,
        )
        mapped_df["Professional_Judgment_Note"] = mapped_df.apply(
            self._professional_judgment_note,
            axis=1,
        )
        return mapped_df

    def _risk_response_level(self, row: pd.Series) -> str:
        risk_level = str(row.get("Risk_Level", ""))
        scope = str(row.get("Guardrail_Adjusted_Scope", row.get("Recommended_Scope", "")))
        severe_findings = int(row.get("Severe_Findings_Count", 0))
        asset_percentage = float(row.get("Asset_Percentage", 0.0))

        if risk_level == "Critical" or scope == "Full Scope":
            return "Significant"
        if risk_level == "High" or severe_findings > 0:
            return "Elevated"
        if risk_level == "Medium" or asset_percentage >= 5.0:
            return "Moderate"
        return "Low"

    def _basis_for_selection(self, row: pd.Series) -> str:
        bases: list[str] = []
        asset_percentage = float(row.get("Asset_Percentage", 0.0))
        risk_level = str(row.get("Risk_Level", ""))
        severe_findings = int(row.get("Severe_Findings_Count", 0))
        scope = str(row.get("Guardrail_Adjusted_Scope", ""))
        reason = str(row.get("Guardrail_Reason", ""))

        if asset_percentage >= 15.0 or scope == "Full Scope":
            bases.append("Materiality")
        if risk_level in {"High", "Critical"}:
            bases.append("Risk")
        if severe_findings > 0:
            bases.append("Prior findings")
        if bool(row.get("Financial_Risk_Guardrail_Applied", False)):
            bases.append("Financial risk")
        if "coverage" in reason.lower():
            bases.append("Coverage")
        if not bases:
            bases.append("Analytical review")
        return " / ".join(bases)

    def _professional_judgment_note(self, row: pd.Series) -> str:
        if bool(row.get("Requires_Human_Review", False)):
            return (
                "Human review is required before finalizing the component scope and work request."
            )
        return (
            "The recommendation is methodology-screened and remains subject to engagement team judgment."
        )
