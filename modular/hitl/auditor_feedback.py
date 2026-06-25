"""Auditor feedback export for future ML improvement."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


FEEDBACK_COLUMNS = [
    "entity_name",
    "ai_recommended_scope",
    "final_auditor_scope",
    "decision_status",
    "auditor_comment",
    "feedback_label",
    "feedback_reason",
    "usable_for_training",
]


def build_auditor_feedback(auditor_review_df: pd.DataFrame) -> pd.DataFrame:
    """Label auditor workpaper decisions for future model-improvement use."""
    missing_columns = [
        column
        for column in [
            "ai_recommended_scope",
            "final_auditor_scope",
            "decision_status",
            "auditor_comment",
        ]
        if column not in auditor_review_df.columns
    ]
    if not _has_entity_column(auditor_review_df):
        missing_columns.append("component_name or entity_name")
    if missing_columns:
        raise ValueError(
            "auditor_review_workpaper.csv is missing required columns: "
            + ", ".join(missing_columns)
        )

    feedback_rows = []
    for _, row in auditor_review_df.fillna("").iterrows():
        ai_scope = _clean(row.get("ai_recommended_scope", ""))
        final_scope = _clean(row.get("final_auditor_scope", ""))
        decision_status = _clean(row.get("decision_status", ""))

        if not final_scope or decision_status.casefold() == "pending":
            feedback_label = "Pending"
            feedback_reason = "Awaiting auditor decision"
            usable_for_training = False
        elif _same_scope(final_scope, ai_scope):
            feedback_label = "Accepted"
            feedback_reason = "Auditor accepted AI recommendation"
            usable_for_training = True
        else:
            feedback_label = "Overridden"
            feedback_reason = "Auditor changed AI recommendation"
            usable_for_training = True

        feedback_rows.append({
            "entity_name": _entity_name(row),
            "ai_recommended_scope": ai_scope,
            "final_auditor_scope": final_scope,
            "decision_status": decision_status,
            "auditor_comment": _clean(row.get("auditor_comment", "")),
            "feedback_label": feedback_label,
            "feedback_reason": feedback_reason,
            "usable_for_training": usable_for_training,
        })

    return pd.DataFrame(feedback_rows, columns=FEEDBACK_COLUMNS)


def build_auditor_feedback_from_final_approval(
    final_scope_df: pd.DataFrame,
    auditor_review_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Label final approved scoping decisions for future model-improvement use."""
    missing_columns = [
        column
        for column in [
            "component_name",
            "ai_recommended_scope",
            "final_auditor_scope",
            "auditor_override",
            "final_decision_source",
        ]
        if column not in final_scope_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "final_approved_scope.csv is missing required columns: "
            + ", ".join(missing_columns)
        )

    comment_lookup = {}
    if auditor_review_df is not None and "auditor_comment" in auditor_review_df.columns:
        comment_lookup = (
            auditor_review_df.fillna("")
            .set_index("component_name")["auditor_comment"]
            .map(_clean)
            .to_dict()
        )

    feedback_rows = []
    for _, row in final_scope_df.fillna("").iterrows():
        component_name = _clean(row["component_name"])
        ai_scope = _clean(row["ai_recommended_scope"])
        final_scope = _clean(row["final_auditor_scope"])
        final_decision_source = _clean(row["final_decision_source"])
        auditor_override = _to_bool(row["auditor_override"])

        if final_decision_source == "ai_default":
            decision_status = "ai_default"
            feedback_label = "AI Default"
            feedback_reason = "No auditor scope provided; AI recommendation used as final default"
            usable_for_training = False
        elif auditor_override:
            decision_status = "modified"
            feedback_label = "Overridden"
            feedback_reason = "Auditor changed AI recommendation"
            usable_for_training = True
        else:
            decision_status = "accepted"
            feedback_label = "Accepted"
            feedback_reason = "Auditor accepted AI recommendation"
            usable_for_training = True

        feedback_rows.append({
            "entity_name": component_name,
            "ai_recommended_scope": ai_scope,
            "final_auditor_scope": final_scope,
            "decision_status": decision_status,
            "auditor_comment": comment_lookup.get(component_name, ""),
            "feedback_label": feedback_label,
            "feedback_reason": feedback_reason,
            "usable_for_training": usable_for_training,
        })

    return pd.DataFrame(feedback_rows, columns=FEEDBACK_COLUMNS)


def generate_auditor_feedback(output_directory: str | Path) -> Path:
    """Create auditor_feedback.csv from final approvals or the auditor workpaper."""
    output_directory = Path(output_directory)
    workpaper_path = output_directory / "auditor_review_workpaper.csv"
    final_scope_path = output_directory / "final_approved_scope.csv"
    feedback_path = output_directory / "auditor_feedback.csv"

    auditor_review_df = pd.read_csv(workpaper_path, dtype=str).fillna("")
    if final_scope_path.is_file():
        final_scope_df = pd.read_csv(final_scope_path, dtype=str).fillna("")
        feedback_df = build_auditor_feedback_from_final_approval(
            final_scope_df,
            auditor_review_df,
        )
    else:
        feedback_df = build_auditor_feedback(auditor_review_df)
    feedback_df.to_csv(feedback_path, index=False)

    return feedback_path


def _clean(value: object) -> str:
    return str(value).strip()


def _same_scope(left: str, right: str) -> bool:
    return left.casefold() == right.casefold()


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() == "true"


def _has_entity_column(df: pd.DataFrame) -> bool:
    return any(column in df.columns for column in ["component_name", "entity_name"])


def _entity_name(row: pd.Series) -> str:
    return _clean(row.get("entity_name", row.get("component_name", "")))
