"""Client data intake loader for completed CSV templates."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

try:
    from .csv_validator import (
        validate_financial_data,
        validate_findings,
        validate_group_entities,
    )
except ImportError:  # pragma: no cover - supports direct manual execution.
    from csv_validator import (  # type: ignore[no-redef]
        validate_financial_data,
        validate_findings,
        validate_group_entities,
    )


ValidationResult = dict[str, object]
Validator = Callable[[str | Path], ValidationResult]


def _clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and string cell values."""
    cleaned = dataframe.copy()
    cleaned.columns = cleaned.columns.astype(str).str.strip()

    string_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    return cleaned


def _validate_csvs(files: dict[str, tuple[Path, Validator]]) -> None:
    """Validate all intake CSV files and raise a combined error if needed."""
    validation_errors: dict[str, tuple[Path, list[str]]] = {}

    for dataset_name, (path, validator) in files.items():
        result = validator(path)
        if result.get("valid"):
            continue

        errors = result.get("errors")
        if not isinstance(errors, list) or not errors:
            errors = ["Unknown validation error"]
        validation_errors[dataset_name] = (path, [str(error) for error in errors])

    if not validation_errors:
        return

    message_lines = ["Client data intake validation failed:"]
    for dataset_name, (path, errors) in validation_errors.items():
        message_lines.append(f"- {dataset_name} ({path}):")
        for error in errors:
            message_lines.append(f"  - {error}")

    raise ValueError("\n".join(message_lines))


def _load_csv(path: Path) -> pd.DataFrame:
    """Load and clean one validated CSV file."""
    return _clean_dataframe(pd.read_csv(path, encoding="utf-8-sig"))


def load_client_data(
    group_entities_path: str | Path,
    findings_path: str | Path,
    financial_data_path: str | Path,
) -> dict[str, pd.DataFrame]:
    """Validate and load completed company intake CSV files.

    Loading accepts local paths today. Future storage integrations, such as
    Azure Blob Storage, should resolve files or streams before calling this
    storage-agnostic loader.
    """
    files: dict[str, tuple[Path, Validator]] = {
        "group_entities": (Path(group_entities_path), validate_group_entities),
        "findings": (Path(findings_path), validate_findings),
        "financial_data": (Path(financial_data_path), validate_financial_data),
    }

    _validate_csvs(files)

    return {
        dataset_name: _load_csv(path)
        for dataset_name, (path, _) in files.items()
    }


if __name__ == "__main__":
    templates_dir = Path(__file__).resolve().parents[2] / "templates"
    dataframes = load_client_data(
        templates_dir / "group_entities_template.csv",
        templates_dir / "findings_template.csv",
        templates_dir / "financial_data_template.csv",
    )

    print("Client data intake validation succeeded.")
    for dataset_name, dataframe in dataframes.items():
        print(f"{dataset_name}: shape={dataframe.shape}")
        print(f"{dataset_name}: columns={list(dataframe.columns)}")
