from __future__ import annotations

import pandas as pd

from modular.bdo.guardrails import BDOGuardrails
from modular.config.config_schema import BDOGuardrailsConfig
from modular.hitl.review_layer import HumanInTheLoopReviewLayer


def _scoped_df(scope: str = "Analytical Procedures") -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Entity": "Component A",
            "Country": "US",
            "Assets": 100.0,
            "Risk_Level": "Low",
            "Asset_Percentage": 1.0,
            "Recommended_Scope": scope,
            "Prediction_Confidence": 0.95,
        }
    ])


def _risks(*risk_types: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "entity_name": "Component A",
            "risk_type": risk_type,
            "risk_description": f"{risk_type} identified",
            "severity": "Medium",
            "source": "test",
            "evidence_value": "fixture",
            "confidence": 1.0,
        }
        for risk_type in risk_types
    ])


def _guarded_row(*risk_types: str, scope: str = "Analytical Procedures") -> pd.Series:
    guarded_df, _ = BDOGuardrails(BDOGuardrailsConfig()).apply(
        _scoped_df(scope),
        identified_risks=_risks(*risk_types),
    )
    return guarded_df.iloc[0]


def test_manual_risk_flag_requires_human_review() -> None:
    row = _guarded_row("Manual Risk Flag", scope="Specific Procedures")

    assert bool(row["Requires_Human_Review"]) is True
    assert bool(row["Financial_Risk_Human_Review_Required"]) is True
    assert "Manual Risk Flag requires human review" in row["Guardrail_Reason"]


def test_liquidity_risk_results_in_at_least_specific_procedures() -> None:
    row = _guarded_row("Liquidity Risk")

    assert row["Guardrail_Adjusted_Scope"] == "Specific Procedures"
    assert row["Recommended_Scope"] == "Specific Procedures"
    assert row["Financial_Risk_Guardrail_Action"] == "Scope adjusted"


def test_high_debt_risk_results_in_at_least_specific_procedures() -> None:
    row = _guarded_row("High Debt Risk")

    assert row["Guardrail_Adjusted_Scope"] == "Specific Procedures"
    assert row["Recommended_Scope"] == "Specific Procedures"
    assert row["Financial_Risk_Guardrail_Action"] == "Scope adjusted"


def test_multiple_financial_risks_require_human_review() -> None:
    row = _guarded_row("Liquidity Risk", "High Debt Risk")

    assert row["Guardrail_Adjusted_Scope"] == "Specific Procedures"
    assert bool(row["Requires_Human_Review"]) is True
    assert bool(row["Financial_Risk_Human_Review_Required"]) is True
    assert "multiple financial risks" in row["Guardrail_Reason"]


def test_guardrail_reasoning_appears_in_review_outputs() -> None:
    guarded_df, _ = BDOGuardrails(BDOGuardrailsConfig()).apply(
        _scoped_df(),
        identified_risks=_risks("Liquidity Risk", "High Debt Risk"),
    )

    artifacts = HumanInTheLoopReviewLayer().prepare(
        guarded_df,
        {"Component A": "Component A explanation."},
    )

    scope_row = guarded_df.iloc[0]
    ai_review_row = artifacts.ai_review_df.iloc[0]
    audit_trail_row = artifacts.audit_trail_df.iloc[0]

    assert "Financial risk guardrail" in scope_row["Guardrail_Reason"]
    assert "Liquidity Risk; High Debt Risk" == scope_row["Financial_Risk_Types"]
    assert ai_review_row["Financial_Risk_Guardrail"] == (
        "Scope adjusted and human review required"
    )
    assert "Liquidity Risk; High Debt Risk" == audit_trail_row["financial_risk_types"]
