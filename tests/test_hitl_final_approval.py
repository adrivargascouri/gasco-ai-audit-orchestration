from __future__ import annotations

import pandas as pd
import pytest

from modular.hitl.final_approval import build_final_approved_scope


def _workpaper(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "component_name": "Component A",
        "ai_recommended_scope": "Specific Procedures",
        "hitl_review_reason": "No mandatory HITL review trigger",
        "financial_risk_guardrail": "None",
        "financial_risk_types": "",
        "financial_risk_guardrail_reason": "",
        "final_auditor_scope": "",
        "decision_status": "pending",
        "auditor_comment": "",
        "approved_by": "",
        "approval_date": "",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_blank_auditor_decision_defaults_to_ai_default() -> None:
    final_scope = build_final_approved_scope(_workpaper([{}]))
    row = final_scope.iloc[0]

    assert row["final_auditor_scope"] == "Specific Procedures"
    assert row["final_decision_source"] == "ai_default"
    assert bool(row["auditor_override"]) is False


def test_auditor_decision_overrides_ai_recommendation() -> None:
    final_scope = build_final_approved_scope(
        _workpaper([{"final_auditor_scope": "Full Scope"}])
    )
    row = final_scope.iloc[0]

    assert row["ai_recommended_scope"] == "Specific Procedures"
    assert row["final_auditor_scope"] == "Full Scope"
    assert row["final_decision_source"] == "auditor"
    assert bool(row["auditor_override"]) is True


def test_invalid_final_auditor_scope_fails_validation() -> None:
    with pytest.raises(ValueError, match="final_auditor_scope"):
        build_final_approved_scope(
            _workpaper([{"final_auditor_scope": "Unsupported Scope"}])
        )


def test_final_decision_source_is_recorded_for_each_row() -> None:
    final_scope = build_final_approved_scope(
        _workpaper([
            {
                "component_name": "Component A",
                "final_auditor_scope": "",
            },
            {
                "component_name": "Component B",
                "final_auditor_scope": "Specific Procedures",
            },
        ])
    )

    source_by_component = dict(
        zip(final_scope["component_name"], final_scope["final_decision_source"])
    )

    assert source_by_component == {
        "Component A": "ai_default",
        "Component B": "auditor",
    }
