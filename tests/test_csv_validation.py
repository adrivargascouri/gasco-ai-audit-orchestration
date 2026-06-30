from __future__ import annotations

from pathlib import Path

from modular.data.company_csv_validator import (
    validate_findings_csv,
    validate_group_structure_csv,
)
from modular.data.csv_validator import validate_financial_data

from .conftest import REPO_ROOT, write_csv


def test_valid_company_templates_pass_validation() -> None:
    templates = REPO_ROOT / "data" / "templates"

    assert validate_group_structure_csv(
        templates / "company_group_structure_template.csv"
    )["valid"]
    assert validate_findings_csv(templates / "company_findings_template.csv")["valid"]
    assert validate_financial_data(
        templates / "company_financial_data_template.csv"
    )["valid"]


def test_missing_required_group_columns_fail(tmp_path: Path) -> None:
    group_file = write_csv(
        tmp_path / "group_missing_revenue.csv",
        ["entity_name", "country", "total_assets", "risk_level"],
        [
            {
                "entity_name": "Component A",
                "country": "US",
                "total_assets": 100,
                "risk_level": "Low",
            }
        ],
    )

    result = validate_group_structure_csv(group_file)

    assert not result["valid"]
    assert any("revenue" in error for error in result["errors"])


def test_invalid_risk_and_severity_values_fail(tmp_path: Path) -> None:
    group_file = write_csv(
        tmp_path / "group_bad_risk.csv",
        ["entity_name", "country", "total_assets", "revenue", "risk_level"],
        [
            {
                "entity_name": "Component A",
                "country": "US",
                "total_assets": 100,
                "revenue": 80,
                "risk_level": "Extreme",
            }
        ],
    )
    findings_file = write_csv(
        tmp_path / "findings_bad_severity.csv",
        ["entity_name", "finding_description", "severity", "year"],
        [
            {
                "entity_name": "Component A",
                "finding_description": "Unusual adjustment",
                "severity": "Severe",
                "year": 2025,
            }
        ],
    )

    group_result = validate_group_structure_csv(group_file)
    findings_result = validate_findings_csv(findings_file)

    assert not group_result["valid"]
    assert any("risk_level" in error for error in group_result["errors"])
    assert not findings_result["valid"]
    assert any("severity" in error for error in findings_result["errors"])


def test_invalid_financial_data_fails_validation(tmp_path: Path) -> None:
    financial_file = write_csv(
        tmp_path / "financial_bad_numeric.csv",
        [
            "entity_name",
            "year",
            "current_assets",
            "current_liabilities",
            "total_assets",
            "total_revenue",
            "net_income",
            "liquidity_ratio",
            "debt_ratio",
        ],
        [
            {
                "entity_name": "Component A",
                "year": 2025,
                "current_assets": 100,
                "current_liabilities": 50,
                "total_assets": 200,
                "total_revenue": 150,
                "net_income": 10,
                "liquidity_ratio": "not-a-number",
                "debt_ratio": 0.4,
            }
        ],
    )

    result = validate_financial_data(financial_file)

    assert not result["valid"]
    assert any("liquidity_ratio" in error for error in result["errors"])
