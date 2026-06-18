"""CSV validation utilities for client intake templates."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


ALLOWED_RISK_LEVELS = {"High", "Medium", "Low"}
ALLOWED_SEVERITIES = {"High", "Medium", "Low"}

REQUIRED_COLUMNS = {
    "group_entities": [
        "entity_name",
        "country",
        "ownership_percentage",
        "currency",
        "assets",
        "revenue",
        "liabilities",
        "employees",
        "risk_level",
        "manual_risk_flag",
    ],
    "findings": [
        "entity_name",
        "finding_description",
        "severity",
        "year",
        "status",
        "department",
    ],
    "financial_data": [
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
}

NUMERIC_FIELDS = {
    "group_entities": [
        "ownership_percentage",
        "assets",
        "revenue",
        "liabilities",
        "employees",
    ],
    "findings": ["year"],
    "financial_data": [
        "year",
        "current_assets",
        "current_liabilities",
        "total_assets",
        "total_revenue",
        "net_income",
        "liquidity_ratio",
        "debt_ratio",
    ],
}


def _result(errors: list[str]) -> dict[str, object]:
    return {"valid": not errors, "errors": errors}


def _is_numeric(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _missing_columns(template_name: str, fieldnames: Iterable[str]) -> list[str]:
    existing_columns = set(fieldnames)
    return [
        column
        for column in REQUIRED_COLUMNS[template_name]
        if column not in existing_columns
    ]


def validate_csv(file_path: str | Path, template_name: str) -> dict[str, object]:
    """Validate a CSV file against one of the client intake template schemas."""
    if template_name not in REQUIRED_COLUMNS:
        return _result([f"Unknown template type: {template_name}"])

    path = Path(file_path)
    errors: list[str] = []

    if not path.is_file():
        return _result([f"Missing CSV file: {path}"])

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames

            if not fieldnames:
                return _result(["Missing header row"])

            for column in _missing_columns(template_name, fieldnames):
                errors.append(f"Missing required column: {column}")

            present_required_columns = [
                column
                for column in REQUIRED_COLUMNS[template_name]
                if column in fieldnames
            ]
            present_numeric_fields = [
                column for column in NUMERIC_FIELDS[template_name] if column in fieldnames
            ]

            for row in reader:
                row_number = reader.line_num

                for column in present_required_columns:
                    value = row.get(column)
                    if value is None or value.strip() == "":
                        errors.append(
                            f"Empty required value in row {row_number}: {column}"
                        )

                for column in present_numeric_fields:
                    value = row.get(column)
                    if value is not None and value.strip() != "" and not _is_numeric(value):
                        errors.append(
                            f"Invalid numeric value in row {row_number}: "
                            f"{column}={value}"
                        )

                if template_name == "group_entities" and "risk_level" in fieldnames:
                    risk_level = (row.get("risk_level") or "").strip()
                    if risk_level and risk_level not in ALLOWED_RISK_LEVELS:
                        errors.append(
                            f"Invalid risk level in row {row_number}: {risk_level}"
                        )

                if template_name == "findings" and "severity" in fieldnames:
                    severity = (row.get("severity") or "").strip()
                    if severity and severity not in ALLOWED_SEVERITIES:
                        errors.append(
                            f"Invalid severity in row {row_number}: {severity}"
                        )
    except csv.Error as exc:
        errors.append(f"Invalid CSV format: {exc}")
    except OSError as exc:
        errors.append(f"Unable to read CSV file: {path} ({exc})")

    return _result(errors)


def validate_group_entities(file_path: str | Path) -> dict[str, object]:
    """Validate a completed group entities intake CSV."""
    return validate_csv(file_path, "group_entities")


def validate_findings(file_path: str | Path) -> dict[str, object]:
    """Validate a completed findings intake CSV."""
    return validate_csv(file_path, "findings")


def validate_financial_data(file_path: str | Path) -> dict[str, object]:
    """Validate a completed financial data intake CSV."""
    return validate_csv(file_path, "financial_data")


def validate_templates(templates_dir: str | Path = "templates") -> dict[str, dict[str, object]]:
    """Validate all client intake template CSV files in a directory."""
    directory = Path(templates_dir)
    return {
        "group_entities": validate_group_entities(
            directory / "group_entities_template.csv"
        ),
        "findings": validate_findings(directory / "findings_template.csv"),
        "financial_data": validate_financial_data(
            directory / "financial_data_template.csv"
        ),
    }


if __name__ == "__main__":
    print(json.dumps(validate_templates("templates"), indent=2))
