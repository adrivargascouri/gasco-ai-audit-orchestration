"""Human-in-the-loop review artifacts for AI-assisted audit scoping."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class HumanReviewArtifacts:
    """Prepared HITL review artifacts before export."""

    ai_review_df: pd.DataFrame
    auditor_review_df: pd.DataFrame
    audit_trail_df: pd.DataFrame
    report_text: str


class HumanInTheLoopReviewLayer:
    """Prepare auditor review workpapers and audit trail records.

    GASCO remains an assistive AI layer: it recommends and explains audit scoping,
    while the auditor must approve, modify, or reject the final scope.
    """

    AI_REVIEW_COLUMNS = [
        "Component_Name",
        "AI_Suggested_Scope",
        "Reasoning",
        "Confidence",
        "Evidence_Used",
        "Guardrails_Triggered",
        "Human_Review_Required",
        "hitl_review_reason",
    ]

    AUDITOR_REVIEW_COLUMNS = [
        "component_name",
        "ai_recommended_scope",
        "hitl_review_reason",
        "final_auditor_scope",
        "decision_status",
        "auditor_comment",
        "approved_by",
        "approval_date",
    ]

    AUDIT_TRAIL_COLUMNS = [
        "component_name",
        "ai_original_recommendation",
        "guardrail_adjusted_recommendation",
        "final_auditor_decision",
        "decision_status",
        "timestamp",
        "reason_for_change",
        "guardrails_triggered",
        "human_review_required",
        "hitl_review_reason",
    ]

    SIGNIFICANT_COMPONENT_THRESHOLD = 15.0
    CONFIDENCE_THRESHOLD = 0.65
    HIGH_RISK_LEVELS = {"High", "Critical"}

    def prepare(
        self,
        scoped_df: pd.DataFrame,
        explanations: dict[str, str],
    ) -> HumanReviewArtifacts:
        """Build all human-review artifacts from guardrail-adjusted AI outputs."""
        timestamp = datetime.now(timezone.utc).isoformat()
        ai_review_rows = []
        auditor_review_rows = []
        audit_trail_rows = []

        for _, row in scoped_df.iterrows():
            component_name = row["Entity"]
            original_scope = row.get("Original_ML_Scope", row["Recommended_Scope"])
            adjusted_scope = row.get("Guardrail_Adjusted_Scope", row["Recommended_Scope"])
            guardrails_triggered = self._guardrails_triggered(row)
            human_review_required = self._human_review_required(row)
            hitl_review_reason = self._hitl_review_reason(row)
            reasoning = self._reasoning(row, explanations.get(component_name, ""))
            evidence_used = self._evidence_used(row)
            decision_status = "pending"

            ai_review_rows.append({
                "Component_Name": component_name,
                "AI_Suggested_Scope": adjusted_scope,
                "Reasoning": reasoning,
                "Confidence": row.get("Prediction_Confidence", ""),
                "Evidence_Used": evidence_used,
                "Guardrails_Triggered": guardrails_triggered,
                "Human_Review_Required": human_review_required,
                "hitl_review_reason": hitl_review_reason,
            })

            auditor_review_rows.append({
                "component_name": component_name,
                "ai_recommended_scope": adjusted_scope,
                "hitl_review_reason": hitl_review_reason,
                "final_auditor_scope": "",
                "decision_status": decision_status,
                "auditor_comment": "",
                "approved_by": "",
                "approval_date": "",
            })

            audit_trail_rows.append({
                "component_name": component_name,
                "ai_original_recommendation": original_scope,
                "guardrail_adjusted_recommendation": adjusted_scope,
                "final_auditor_decision": "",
                "decision_status": decision_status,
                "timestamp": timestamp,
                "reason_for_change": self._reason_for_change(row),
                "guardrails_triggered": guardrails_triggered,
                "human_review_required": human_review_required,
                "hitl_review_reason": hitl_review_reason,
            })

        ai_review_df = pd.DataFrame(ai_review_rows, columns=self.AI_REVIEW_COLUMNS)
        auditor_review_df = pd.DataFrame(
            auditor_review_rows,
            columns=self.AUDITOR_REVIEW_COLUMNS,
        )
        audit_trail_df = pd.DataFrame(audit_trail_rows, columns=self.AUDIT_TRAIL_COLUMNS)

        return HumanReviewArtifacts(
            ai_review_df=ai_review_df,
            auditor_review_df=auditor_review_df,
            audit_trail_df=audit_trail_df,
            report_text=self._report(scoped_df, auditor_review_df, audit_trail_df),
        )

    def export(self, artifacts: HumanReviewArtifacts, output_directory: str | Path) -> dict[str, Path]:
        """Write HITL artifacts to the configured output directory."""
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        paths = {
            "human_review_ai_recommendations": output_directory / "human_review_ai_recommendations.csv",
            "auditor_review_workpaper": output_directory / "auditor_review_workpaper.csv",
            "audit_trail": output_directory / "audit_trail.csv",
            "final_human_review_report": output_directory / "final_human_review_report.txt",
        }
        artifacts.ai_review_df.to_csv(paths["human_review_ai_recommendations"], index=False)
        artifacts.auditor_review_df.to_csv(paths["auditor_review_workpaper"], index=False)
        artifacts.audit_trail_df.to_csv(paths["audit_trail"], index=False)
        paths["final_human_review_report"].write_text(
            artifacts.report_text,
            encoding="utf-8",
        )
        return paths

    def _reasoning(self, row: pd.Series, explanation: str) -> str:
        reason = str(row.get("Guardrail_Reason", "")).strip()
        short_explanation = " ".join(
            line[2:] for line in explanation.splitlines() if line.startswith("- ")
        )
        parts = []
        if short_explanation:
            parts.append(short_explanation)
        if reason:
            parts.append(f"Guardrail assessment: {reason}")
        return " ".join(parts)

    def _evidence_used(self, row: pd.Series) -> str:
        evidence = [
            f"Assets={row.get('Assets')}",
            f"Asset_Percentage={row.get('Asset_Percentage')}",
            f"Risk_Level={row.get('Risk_Level')}",
            f"Severe_Findings_Count={row.get('Severe_Findings_Count', 0)}",
            f"Prediction_Confidence={row.get('Prediction_Confidence')}",
        ]
        basis = row.get("Basis_For_Selection", "")
        if basis:
            evidence.append(f"Basis_For_Selection={basis}")
        return "; ".join(evidence)

    def _guardrails_triggered(self, row: pd.Series) -> str:
        action = str(row.get("Guardrail_Action", "Accepted"))
        reason = str(row.get("Guardrail_Reason", "")).strip()
        if action == "Accepted":
            return "None"
        return reason or action

    def _human_review_required(self, row: pd.Series) -> bool:
        return bool(self._review_triggers(row))

    def _review_triggers(self, row: pd.Series) -> list[str]:
        risk_level = str(row.get("Risk_Level", ""))
        asset_percentage = float(row.get("Asset_Percentage", 0.0))
        confidence = float(row.get("Prediction_Confidence", 0.0))
        original_scope = row.get("Original_ML_Scope", row["Recommended_Scope"])
        adjusted_scope = row.get("Guardrail_Adjusted_Scope", row["Recommended_Scope"])
        guardrail_adjusted = original_scope != adjusted_scope

        triggers = []
        if confidence < self.CONFIDENCE_THRESHOLD:
            triggers.append("Low ML confidence")
        if risk_level in self.HIGH_RISK_LEVELS:
            triggers.append("High risk component")
        if asset_percentage >= self.SIGNIFICANT_COMPONENT_THRESHOLD:
            triggers.append("Significant component")
        if guardrail_adjusted:
            triggers.append("Guardrail adjusted AI recommendation")
        if bool(row.get("Requires_Human_Review", False)) and not triggers:
            triggers.append("Legacy guardrail review flag")
        return triggers

    def _hitl_review_reason(self, row: pd.Series) -> str:
        triggers = self._review_triggers(row)
        if not triggers:
            return "No mandatory HITL review trigger"
        if len(triggers) > 1:
            return f"Multiple review triggers: {'; '.join(triggers)}"
        return triggers[0]

    def _reason_for_change(self, row: pd.Series) -> str:
        original_scope = row.get("Original_ML_Scope", row["Recommended_Scope"])
        adjusted_scope = row.get("Guardrail_Adjusted_Scope", row["Recommended_Scope"])
        reason = str(row.get("Guardrail_Reason", "")).strip()
        if original_scope != adjusted_scope:
            return reason or f"Guardrails changed scope from {original_scope} to {adjusted_scope}."
        if bool(row.get("Requires_Human_Review", False)):
            return reason or "Human review required by configured guardrails."
        return "No guardrail change. Awaiting auditor decision."

    def _report(
        self,
        scoped_df: pd.DataFrame,
        auditor_review_df: pd.DataFrame,
        audit_trail_df: pd.DataFrame,
    ) -> str:
        review_required_count = int(audit_trail_df["human_review_required"].sum())
        adjusted_count = int(
            (
                scoped_df["Original_ML_Scope"]
                != scoped_df["Guardrail_Adjusted_Scope"]
            ).sum()
        )
        pending_count = int((auditor_review_df["decision_status"] == "pending").sum())

        return "\n".join([
            "GASCO Human-in-the-Loop AI Review Report",
            "",
            "Purpose",
            "This report documents the Human-in-the-Loop AI Review Layer for audit scoping decisions.",
            "",
            "Core principle",
            "GASCO only suggests audit scoping decisions.",
            "The auditor makes the final decision.",
            "",
            "Human review policy",
            "Human review is required for low confidence, high risk, significant components, or guardrail adjustments.",
            "The hitl_review_reason field documents the stricter HITL policy and may differ from the legacy Requires_Human_Review guardrail flag.",
            "",
            "Current review status",
            f"Components prepared for auditor review: {len(scoped_df)}.",
            f"Components requiring human review: {review_required_count}.",
            f"Recommendations adjusted by guardrails: {adjusted_count}.",
            f"Pending final auditor decisions: {pending_count}.",
            "",
            "Workflow",
            "AI Recommendation -> Auditor Review -> Final Auditor Decision -> Audit Trail",
            "",
            "Auditor workpaper",
            "Auditors should complete final_auditor_scope, decision_status, auditor_comment, approved_by, and approval_date before finalizing scoping.",
            "Allowed decision_status values: pending, accepted, modified, rejected.",
            "",
            "Audit trail",
            "The audit_trail.csv file retains the original AI recommendation, guardrail-adjusted recommendation, final auditor decision fields, timestamp, and reason for change.",
        ])
