"""Data loading utilities."""
from pathlib import Path
from typing import Tuple
import pandas as pd
from ..domain.models import Entity, Finding
from .company_csv_validator import (
    load_company_findings,
    load_company_group_structure,
)


class DataLoader:
    """Loads group structure and findings data."""

    @staticmethod
    def load_group_data(
        group_path: str | Path = "data/group_structure.json",
        findings_path: str | Path = "data/findings_repo.csv",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load group structure and findings data.

        Args:
            group_path: Path to group structure JSON file
            findings_path: Path to findings repository CSV file

        Returns:
            Tuple of (group_df, findings_df)
        """
        group_path = Path(group_path)
        group_df = (
            load_company_group_structure(group_path)
            if group_path.suffix.lower() == ".csv"
            else pd.read_json(str(group_path))
        )
        findings_df = DataLoader.load_findings_dataframe(findings_path)
        return group_df, findings_df

    @staticmethod
    def load_findings_dataframe(findings_path: str | Path) -> pd.DataFrame:
        """Load either the internal findings CSV or a company findings CSV."""
        dataframe = pd.read_csv(str(findings_path), encoding="utf-8-sig")
        normalized_columns = {str(column).strip() for column in dataframe.columns}
        if "entity_name" in normalized_columns:
            return load_company_findings(findings_path)
        return dataframe

    @staticmethod
    def load_entities(group_path: str | Path = "data/group_structure.json") -> list[Entity]:
        """Load entities as domain model objects.

        Args:
            group_path: Path to group structure JSON file

        Returns:
            List of Entity objects
        """
        df = pd.read_json(str(group_path))
        entities = []
        for _, row in df.iterrows():
            entity = Entity(
                entity=row["Entity"],
                country=row["Country"],
                revenue=row["Revenue"],
                assets=row["Assets"],
                risk_level=row["Risk_Level"],
            )
            entities.append(entity)
        return entities

    @staticmethod
    def load_findings(
        findings_path: str | Path = "data/findings_repo.csv",
    ) -> list[Finding]:
        """Load findings as domain model objects.

        Args:
            findings_path: Path to findings repository CSV file

        Returns:
            List of Finding objects
        """
        df = pd.read_csv(str(findings_path))
        findings = []
        for _, row in df.iterrows():
            finding = Finding(
                entity=row["Entity"],
                finding=row["Finding"],
                severity=row["Severity"],
            )
            findings.append(finding)
        return findings
