from __future__ import annotations

import pandas as pd

from modular.agents.risk_discovery_agent import discover_risks


def test_low_liquidity_produces_liquidity_risk() -> None:
    risks = discover_risks({
        "financial_data": pd.DataFrame([
            {
                "entity_name": "Component A",
                "total_assets": 100,
                "liquidity_ratio": 1.1,
                "debt_ratio": 0.2,
            }
        ])
    })

    assert "Liquidity Risk" in set(risks["risk_type"])


def test_high_debt_produces_high_debt_risk() -> None:
    risks = discover_risks({
        "financial_data": pd.DataFrame([
            {
                "entity_name": "Component A",
                "total_assets": 100,
                "liquidity_ratio": 2.0,
                "debt_ratio": 0.7,
            }
        ])
    })

    assert "High Debt Risk" in set(risks["risk_type"])


def test_manual_risk_flag_produces_manual_risk_flag() -> None:
    risks = discover_risks({
        "group_entities": pd.DataFrame([
            {
                "entity_name": "Component A",
                "manual_risk_flag": "Yes",
            }
        ])
    })

    assert "Manual Risk Flag" in set(risks["risk_type"])


def test_high_findings_are_treated_as_severe() -> None:
    risks = discover_risks({
        "findings": pd.DataFrame([
            {
                "entity_name": "Component A",
                "finding_description": "Material control gap",
                "severity": "High",
            }
        ])
    })

    assert "High Finding Risk" in set(risks["risk_type"])


def test_critical_findings_are_treated_as_severe() -> None:
    risks = discover_risks({
        "findings": pd.DataFrame([
            {
                "entity_name": "Component A",
                "finding_description": "Critical reporting gap",
                "severity": "Critical",
            }
        ])
    })

    assert "High Finding Risk" in set(risks["risk_type"])
