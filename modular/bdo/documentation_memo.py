"""Generate BDO-style scoping documentation memo."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class BDODocumentationMemo:
    """Write a concise audit trail memo using original wording."""

    def generate(
        self,
        scoped_df: pd.DataFrame,
        coverage_summary: dict,
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        adjusted_count = int((scoped_df["Original_ML_Scope"] != scoped_df["Guardrail_Adjusted_Scope"]).sum())
        review_df = scoped_df[scoped_df["Requires_Human_Review"] == True]
        distribution = scoped_df["Guardrail_Adjusted_Scope"].value_counts().to_dict()
        ml_distribution = scoped_df["Original_ML_Scope"].value_counts().to_dict()
        financial_guardrail_count = (
            int(scoped_df["Financial_Risk_Guardrail_Applied"].sum())
            if "Financial_Risk_Guardrail_Applied" in scoped_df.columns
            else 0
        )

        lines = [
            "GASCO BDO-Style Scoping Documentation Memo",
            "",
            "Purpose",
            "This memo documents the ML-assisted component scope recommendation and the methodology guardrails applied before final review.",
            "",
            "ML-assisted scope recommendation",
            f"The model produced recommendations for {len(scoped_df)} components. Original ML distribution: {ml_distribution}.",
            "",
            "Guardrail adjustments",
            f"Guardrail-adjusted distribution: {distribution}. Adjusted recommendations: {adjusted_count}.",
            f"Financial-risk guardrail triggers documented: {financial_guardrail_count}.",
        ]

        for _, row in scoped_df.iterrows():
            if row["Guardrail_Action"] != "Accepted":
                lines.append(
                    f"- {row['Entity']}: {row['Original_ML_Scope']} -> {row['Guardrail_Adjusted_Scope']} ({row['Guardrail_Reason']})"
                )

        lines.extend([
            "",
            "Basis for component selection",
        ])
        for _, row in scoped_df.iterrows():
            lines.append(
                f"- {row['Entity']}: {row['Basis_For_Selection']} | selected for work: {row['Component_Selected_For_Work']}."
            )

        lines.extend([
            "",
            "Coverage result",
            (
                f"Adjusted coverage is {coverage_summary['Coverage_Percentage']:.2f}% "
                f"({coverage_summary['Covered_Assets']:.0f} of {coverage_summary['Total_Assets']:.0f} assets). "
                f"Status: {coverage_summary['Status']}."
            ),
            "",
            "Components requiring human review",
        ])
        if review_df.empty:
            lines.append("No components were flagged for mandatory human review by the configured guardrails.")
        else:
            for _, row in review_df.iterrows():
                lines.append(f"- {row['Entity']}: {row['Guardrail_Reason']}")

        lines.extend([
            "",
            "Professional judgment reminder",
            "The ML model is an assistive input. Final scoping decisions require engagement team evaluation, partner oversight where relevant, and documentation of professional judgment.",
            "",
            "Audit trail note",
            "The original ML recommendation, adjusted recommendation, guardrail action, reason, and review flag are retained in the exported scoping files.",
        ])

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
