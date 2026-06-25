"""Final HITL approval workflow for auditor-approved scoping."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from modular.hitl.auditor_feedback import build_auditor_feedback_from_final_approval


ALLOWED_SCOPES = (
    "Full Scope",
    "Specific Procedures",
    "Analytical Procedures",
)
FINAL_APPROVED_SCOPE_COLUMNS = [
    "component_name",
    "ai_recommended_scope",
    "final_auditor_scope",
    "auditor_override",
    "final_decision_source",
]
REQUIRED_WORKPAPER_COLUMNS = [
    "component_name",
    "ai_recommended_scope",
    "hitl_review_reason",
    "final_auditor_scope",
]


def run_final_approval_workflow(output_directory: str | Path) -> dict[str, Path]:
    """Generate final approved scope and refresh HITL downstream artifacts."""
    output_directory = Path(output_directory)
    workpaper_path = output_directory / "auditor_review_workpaper.csv"
    final_scope_path = output_directory / "final_approved_scope.csv"
    audit_trail_path = output_directory / "audit_trail.csv"
    feedback_path = output_directory / "auditor_feedback.csv"

    workpaper_df = _read_workpaper(workpaper_path)
    final_scope_df = build_final_approved_scope(workpaper_df)
    final_scope_df.to_csv(final_scope_path, index=False)

    update_audit_trail(audit_trail_path, final_scope_df)

    feedback_df = build_auditor_feedback_from_final_approval(
        final_scope_df,
        workpaper_df,
    )
    feedback_df.to_csv(feedback_path, index=False)

    return {
        "final_approved_scope": final_scope_path,
        "audit_trail": audit_trail_path,
        "auditor_feedback": feedback_path,
    }


def build_final_approved_scope(workpaper_df: pd.DataFrame) -> pd.DataFrame:
    """Build final auditor-approved scopes from the auditor workpaper."""
    _validate_required_columns(workpaper_df)
    _validate_unique_components(workpaper_df)

    rows = []
    for index, row in workpaper_df.fillna("").iterrows():
        row_number = index + 2
        component_name = _clean(row["component_name"])
        ai_scope = _normalize_scope(row["ai_recommended_scope"], row_number, "ai_recommended_scope")
        raw_final_scope = _clean(row["final_auditor_scope"])
        final_scope_was_filled = bool(raw_final_scope)
        final_scope = (
            _normalize_scope(raw_final_scope, row_number, "final_auditor_scope")
            if final_scope_was_filled
            else ai_scope
        )

        if not component_name:
            raise ValueError(f"Row {row_number}: component_name must not be empty.")

        rows.append({
            "component_name": component_name,
            "ai_recommended_scope": ai_scope,
            "final_auditor_scope": final_scope,
            "auditor_override": final_scope != ai_scope,
            "final_decision_source": (
                "auditor" if final_scope_was_filled else "ai_default"
            ),
        })

    return pd.DataFrame(rows, columns=FINAL_APPROVED_SCOPE_COLUMNS)


def update_audit_trail(
    audit_trail_path: str | Path,
    final_scope_df: pd.DataFrame,
) -> Path:
    """Update audit trail rows with final approved scoping decisions."""
    audit_trail_path = Path(audit_trail_path)
    if not audit_trail_path.is_file():
        raise FileNotFoundError(f"Missing audit trail file: {audit_trail_path}")

    audit_trail_df = pd.read_csv(audit_trail_path, dtype=str).fillna("")
    if "component_name" not in audit_trail_df.columns:
        raise ValueError("audit_trail.csv is missing required column: component_name")

    final_scope_lookup = final_scope_df.set_index("component_name")
    missing_components = sorted(
        set(audit_trail_df["component_name"]) - set(final_scope_lookup.index)
    )
    if missing_components:
        raise ValueError(
            "Final approved scope is missing audit trail component(s): "
            + ", ".join(missing_components)
        )

    audit_trail_df["final_auditor_decision"] = audit_trail_df["component_name"].map(
        final_scope_lookup["final_auditor_scope"]
    )
    audit_trail_df["decision_status"] = audit_trail_df["component_name"].map(
        final_scope_lookup.apply(_audit_trail_decision_status, axis=1)
    )

    # Appended columns preserve the existing audit-trail schema while documenting
    # how each final decision was reached.
    audit_trail_df["auditor_override"] = audit_trail_df["component_name"].map(
        final_scope_lookup["auditor_override"]
    )
    audit_trail_df["final_decision_source"] = audit_trail_df["component_name"].map(
        final_scope_lookup["final_decision_source"]
    )

    audit_trail_df.to_csv(audit_trail_path, index=False)
    return audit_trail_path


def _read_workpaper(workpaper_path: Path) -> pd.DataFrame:
    if not workpaper_path.is_file():
        raise FileNotFoundError(f"Missing auditor review workpaper: {workpaper_path}")
    return pd.read_csv(workpaper_path, dtype=str).fillna("")


def _validate_required_columns(workpaper_df: pd.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_WORKPAPER_COLUMNS if column not in workpaper_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "auditor_review_workpaper.csv is missing required columns: "
            + ", ".join(missing_columns)
        )


def _validate_unique_components(workpaper_df: pd.DataFrame) -> None:
    component_names = workpaper_df["component_name"].fillna("").map(_clean)
    duplicates = sorted(component_names[component_names.duplicated()].unique())
    duplicates = [component for component in duplicates if component]
    if duplicates:
        raise ValueError(
            "auditor_review_workpaper.csv contains duplicate component_name values: "
            + ", ".join(duplicates)
        )


def _normalize_scope(value: object, row_number: int, column_name: str) -> str:
    cleaned_value = _clean(value)
    scope_by_key = {scope.casefold(): scope for scope in ALLOWED_SCOPES}
    normalized_scope = scope_by_key.get(cleaned_value.casefold())
    if normalized_scope:
        return normalized_scope

    accepted = ", ".join(ALLOWED_SCOPES)
    raise ValueError(
        f"Row {row_number}: {column_name} must be one of {accepted}; "
        f"received '{cleaned_value}'."
    )


def _audit_trail_decision_status(row: pd.Series) -> str:
    if row["final_decision_source"] == "ai_default":
        return "ai_default"
    if _to_bool(row["auditor_override"]):
        return "modified"
    return "accepted"


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() == "true"


def _clean(value: object) -> str:
    return str(value).strip()
