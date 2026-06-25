"""Validation and normalization for company-provided GASCO CSV files."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


GROUP_REQUIRED_COLUMNS = (
    "entity_name",
    "country",
    "total_assets",
    "revenue",
    "risk_level",
)
FINDINGS_REQUIRED_COLUMNS = (
    "entity_name",
    "finding_description",
    "severity",
    "year",
)
ALLOWED_LEVELS = ("Low", "Medium", "High", "Critical")


def _result(errors: list[str]) -> dict[str, object]:
    return {"valid": not errors, "errors": errors}


def _read_csv(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    if not path.is_file():
        return None, [f"File not found: {path}"]
    try:
        dataframe = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    except pd.errors.EmptyDataError:
        return None, [f"CSV file has no header row: {path}"]
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        return None, [f"Could not read CSV file '{path}': {exc}"]

    dataframe.columns = dataframe.columns.astype(str).str.strip()
    if dataframe.columns.duplicated().any():
        duplicates = sorted(set(dataframe.columns[dataframe.columns.duplicated()]))
        return None, [f"Duplicate column name(s): {', '.join(duplicates)}"]
    return dataframe, []


def _validate_columns(dataframe: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    missing = [column for column in required if column not in dataframe.columns]
    if not missing:
        return []
    return [f"Missing required column(s): {', '.join(missing)}"]


def _value(row: pd.Series, column: str) -> str:
    return str(row[column]).strip()


def _validate_level(value: str, row_number: int, column: str) -> list[str]:
    if value in ALLOWED_LEVELS:
        return []
    accepted = ", ".join(ALLOWED_LEVELS)
    return [
        f"Row {row_number}: {column} must be one of {accepted}; received '{value}'."
    ]


def validate_group_structure_csv(file_path: str | Path) -> dict[str, object]:
    """Validate a company group structure CSV and return all user-facing errors."""
    dataframe, errors = _read_csv(Path(file_path))
    if dataframe is None:
        return _result(errors)

    errors.extend(_validate_columns(dataframe, GROUP_REQUIRED_COLUMNS))
    if errors:
        return _result(errors)

    for index, row in dataframe.iterrows():
        row_number = index + 2
        for column in ("entity_name", "country"):
            if not _value(row, column):
                errors.append(f"Row {row_number}: {column} must not be empty.")

        for column in ("total_assets", "revenue"):
            raw_value = _value(row, column)
            try:
                numeric_value = float(raw_value)
                if not math.isfinite(numeric_value) or numeric_value < 0:
                    raise ValueError
            except ValueError:
                errors.append(
                    f"Row {row_number}: {column} must be a non-negative number; "
                    f"received '{raw_value}'."
                )

        errors.extend(_validate_level(_value(row, "risk_level"), row_number, "risk_level"))

    return _result(errors)


def validate_findings_csv(file_path: str | Path) -> dict[str, object]:
    """Validate an optional company findings CSV and return all user-facing errors."""
    dataframe, errors = _read_csv(Path(file_path))
    if dataframe is None:
        return _result(errors)

    errors.extend(_validate_columns(dataframe, FINDINGS_REQUIRED_COLUMNS))
    if errors:
        return _result(errors)

    for index, row in dataframe.iterrows():
        row_number = index + 2
        for column in ("entity_name", "finding_description"):
            if not _value(row, column):
                errors.append(f"Row {row_number}: {column} must not be empty.")

        errors.extend(_validate_level(_value(row, "severity"), row_number, "severity"))
        year = _value(row, "year")
        if year:
            try:
                int(year)
            except ValueError:
                errors.append(
                    f"Row {row_number}: year must be a whole number when provided; "
                    f"received '{year}'."
                )

    return _result(errors)


def _raise_for_errors(path: str | Path, result: dict[str, object]) -> None:
    if result["valid"]:
        return
    details = "\n".join(f"- {error}" for error in result["errors"])
    raise ValueError(f"Company CSV validation failed for '{path}':\n{details}")


def load_company_group_structure(file_path: str | Path) -> pd.DataFrame:
    """Load a validated company group CSV in the pipeline's internal format."""
    result = validate_group_structure_csv(file_path)
    _raise_for_errors(file_path, result)
    dataframe = pd.read_csv(file_path, encoding="utf-8-sig")
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    dataframe = dataframe.rename(columns={
        "entity_name": "Entity",
        "country": "Country",
        "total_assets": "Assets",
        "revenue": "Revenue",
        "risk_level": "Risk_Level",
    })
    columns = ["Entity", "Country", "Assets", "Revenue", "Risk_Level"]
    if "manual_risk_flag" in dataframe.columns:
        columns.append("manual_risk_flag")
    return dataframe[columns]


def load_company_findings(file_path: str | Path) -> pd.DataFrame:
    """Load a validated company findings CSV in the pipeline's internal format."""
    result = validate_findings_csv(file_path)
    _raise_for_errors(file_path, result)
    dataframe = pd.read_csv(file_path, encoding="utf-8-sig")
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    dataframe = dataframe.rename(columns={
        "entity_name": "Entity",
        "finding_description": "Finding",
        "severity": "Severity",
        "year": "Year",
    })
    return dataframe[["Entity", "Finding", "Severity", "Year"]]
