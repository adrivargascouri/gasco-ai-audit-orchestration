"""Rule-based risk discovery agent for validated client intake data."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd


RISK_COLUMNS = [
    "entity_name",
    "risk_type",
    "risk_description",
    "severity",
    "source",
    "evidence_value",
    "confidence",
]
SEVERE_FINDING_LEVELS = {"high", "critical"}
FINANCIAL_GUARDRAIL_RISK_TYPES = (
    "Manual Risk Flag",
    "Liquidity Risk",
    "High Debt Risk",
)


class RiskDiscoveryAgent:
    """Identify pre-scope audit risks from validated company data."""

    def discover(self, client_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Run deterministic risk discovery rules over client intake data."""
        risks: list[dict[str, Any]] = []

        financial_data = client_data.get("financial_data", pd.DataFrame())
        findings = client_data.get("findings", pd.DataFrame())
        group_entities = client_data.get("group_entities", pd.DataFrame())

        risks.extend(self._discover_financial_risks(financial_data))
        risks.extend(self._discover_finding_risks(findings))
        risks.extend(self._discover_manual_flags(group_entities))

        return pd.DataFrame(risks, columns=RISK_COLUMNS)

    def _discover_financial_risks(
        self, financial_data: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Discover liquidity, debt, and significant component risks."""
        if financial_data.empty:
            return []

        financial_df = financial_data.copy()
        for column in [
            "liquidity_ratio",
            "debt_ratio",
            "total_assets",
            "asset_percentage",
        ]:
            if column in financial_df.columns:
                financial_df[column] = pd.to_numeric(financial_df[column], errors="coerce")

        risks: list[dict[str, Any]] = []
        total_group_assets = financial_df["total_assets"].sum()

        for _, row in financial_df.iterrows():
            entity_name = row["entity_name"]
            year_detail = self._year_detail(row)

            liquidity_ratio = row.get("liquidity_ratio")
            if pd.notna(liquidity_ratio) and liquidity_ratio < 1.2:
                severity = "High" if liquidity_ratio < 1.0 else "Medium"
                risks.append(
                    self._risk_row(
                        entity_name=entity_name,
                        risk_type="Liquidity Risk",
                        risk_description=(
                            f"Liquidity ratio is below 1.2{year_detail}."
                        ),
                        severity=severity,
                        source="financial_data",
                        evidence_value=round(float(liquidity_ratio), 4),
                    )
                )

            debt_ratio = row.get("debt_ratio")
            if pd.notna(debt_ratio) and debt_ratio > 0.6:
                severity = "High" if debt_ratio > 0.75 else "Medium"
                risks.append(
                    self._risk_row(
                        entity_name=entity_name,
                        risk_type="High Debt Risk",
                        risk_description=f"Debt ratio is above 0.6{year_detail}.",
                        severity=severity,
                        source="financial_data",
                        evidence_value=round(float(debt_ratio), 4),
                    )
                )

            assets_percentage = row.get("asset_percentage")
            if pd.isna(assets_percentage):
                total_assets = row.get("total_assets")
                if total_group_assets > 0 and pd.notna(total_assets):
                    assets_percentage = total_assets / total_group_assets

            if pd.notna(assets_percentage) and assets_percentage >= 0.15:
                risks.append(
                    self._risk_row(
                        entity_name=entity_name,
                        risk_type="Significant Component Risk",
                        risk_description=(
                            "Entity assets are at least 15% of total group assets"
                            f"{year_detail}."
                        ),
                        severity="High",
                        source="financial_data",
                        evidence_value=round(float(assets_percentage), 4),
                    )
                )

        return risks

    def _discover_finding_risks(self, findings: pd.DataFrame) -> list[dict[str, Any]]:
        """Discover risks from high-severity findings."""
        if findings.empty:
            return []

        risks: list[dict[str, Any]] = []
        high_findings = findings[
            findings["severity"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .isin(SEVERE_FINDING_LEVELS)
        ]

        for _, row in high_findings.iterrows():
            risks.append(
                self._risk_row(
                    entity_name=row["entity_name"],
                    risk_type="High Finding Risk",
                    risk_description=self._finding_description(row),
                    severity="High",
                    source="findings",
                    evidence_value="High",
                )
            )

        return risks

    def _discover_manual_flags(
        self, group_entities: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Discover risks from manually flagged entities."""
        if group_entities.empty:
            return []

        risks: list[dict[str, Any]] = []
        flagged_entities = group_entities[
            group_entities["manual_risk_flag"].astype(str).str.strip().str.casefold()
            == "yes"
        ]

        for _, row in flagged_entities.iterrows():
            risks.append(
                self._risk_row(
                    entity_name=row["entity_name"],
                    risk_type="Manual Risk Flag",
                    risk_description="Entity was manually flagged for risk review.",
                    severity="Medium",
                    source="group_entities",
                    evidence_value="Yes",
                )
            )

        return risks

    def _risk_row(
        self,
        *,
        entity_name: str,
        risk_type: str,
        risk_description: str,
        severity: str,
        source: str,
        evidence_value: Any,
    ) -> dict[str, Any]:
        """Build one standardized risk record."""
        return {
            "entity_name": entity_name,
            "risk_type": risk_type,
            "risk_description": risk_description,
            "severity": severity,
            "source": source,
            "evidence_value": evidence_value,
            "confidence": 1.0,
        }

    def _year_detail(self, row: pd.Series) -> str:
        """Return a compact year note when financial data includes year."""
        year = row.get("year")
        if pd.isna(year):
            return ""

        return f" for {int(year)}"

    def _finding_description(self, row: pd.Series) -> str:
        """Return a finding risk description using available finding detail."""
        description = str(row.get("finding_description", "")).strip()
        if not description:
            return "High-severity finding identified."

        return f"High-severity finding identified: {description}"


def discover_risks(client_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Discover potential audit risks from validated client intake data."""
    return RiskDiscoveryAgent().discover(client_data)


def build_official_risk_discovery_input(
    group_df: pd.DataFrame,
    findings_df: pd.DataFrame,
    financial_file: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Adapt official pipeline dataframes to the risk discovery agent input shape."""
    financial_data, financial_status = _aligned_financial_data(
        group_df,
        financial_file,
    )

    findings = pd.DataFrame({
        "entity_name": findings_df.get("Entity", pd.Series(dtype=object)),
        "finding_description": findings_df.get("Finding", pd.Series(dtype=object)),
        "severity": findings_df.get("Severity", pd.Series(dtype=object)),
    })

    group_entities = pd.DataFrame({
        "entity_name": group_df.get("Entity", pd.Series(dtype=object)),
        "manual_risk_flag": group_df.get(
            "manual_risk_flag",
            pd.Series(pd.NA, index=group_df.index),
        ),
    })
    if "manual_risk_flag" in financial_data.columns:
        financial_flags = (
            financial_data[["entity_name", "manual_risk_flag"]]
            .dropna(subset=["manual_risk_flag"])
            .drop_duplicates(subset=["entity_name"])
        )
        group_entities = group_entities.merge(
            financial_flags,
            on="entity_name",
            how="left",
            suffixes=("", "_financial"),
        )
        group_entities["manual_risk_flag"] = group_entities[
            "manual_risk_flag_financial"
        ].combine_first(group_entities["manual_risk_flag"])
        group_entities = group_entities.drop(columns=["manual_risk_flag_financial"])

    return (
        {
            "financial_data": financial_data,
            "findings": findings,
            "group_entities": group_entities,
        },
        financial_status,
    )


def _clean_csv_dataframe(path: Path) -> pd.DataFrame:
    """Load a validated CSV and strip whitespace from labels and text cells."""
    dataframe = pd.read_csv(path, encoding="utf-8-sig")
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    string_columns = dataframe.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        dataframe[column] = dataframe[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return dataframe


def _entity_key(values: pd.Series) -> pd.Series:
    """Normalize entity labels for component-name alignment."""
    return values.astype(str).str.strip().str.casefold()


def _base_financial_data(group_df: pd.DataFrame) -> pd.DataFrame:
    """Build the group-derived financial frame used when no financial CSV is used."""
    total_assets = pd.to_numeric(
        group_df.get("Assets", pd.Series(dtype=float)),
        errors="coerce",
    )
    financial_data = pd.DataFrame({
        "entity_name": group_df.get("Entity", pd.Series(dtype=object)),
        "total_assets": total_assets,
    })
    total_group_assets = total_assets.sum()
    financial_data["asset_percentage"] = (
        total_assets / total_group_assets if total_group_assets > 0 else pd.NA
    )
    return financial_data


def _aligned_financial_data(
    group_df: pd.DataFrame,
    financial_file: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Align optional financial intake rows to official group entities."""
    base_financial_data = _base_financial_data(group_df)
    if financial_file is None:
        return base_financial_data, {
            "provided": False,
            "used": False,
            "message": (
                "Financial data: no --financial-file provided; risk discovery "
                "used group assets only."
            ),
        }

    financial_df = _clean_csv_dataframe(financial_file)
    financial_df["_entity_key"] = _entity_key(financial_df["entity_name"])

    aligned = base_financial_data.copy()
    aligned["_entity_key"] = _entity_key(aligned["entity_name"])
    aligned = aligned.merge(
        financial_df,
        on="_entity_key",
        how="left",
        suffixes=("", "_financial"),
    )

    matched_mask = aligned.get("entity_name_financial", pd.Series(dtype=object)).notna()
    matched_rows = int(matched_mask.sum())
    matched_entities = int(aligned.loc[matched_mask, "_entity_key"].nunique())
    group_keys = set(aligned["_entity_key"].dropna())
    ignored_rows = int((~financial_df["_entity_key"].isin(group_keys)).sum())

    if matched_rows == 0:
        return base_financial_data, {
            "provided": True,
            "used": False,
            "path": financial_file,
            "matched_rows": 0,
            "matched_entities": 0,
            "ignored_rows": ignored_rows,
            "message": (
                f"Financial data: provided at {financial_file}, but no entity "
                "names matched the loaded group structure; risk discovery used "
                "group assets only."
            ),
        }

    aligned["total_assets"] = pd.to_numeric(
        aligned.get("total_assets_financial", aligned["total_assets"]),
        errors="coerce",
    ).fillna(pd.to_numeric(aligned["total_assets"], errors="coerce"))
    total_group_assets = pd.to_numeric(
        base_financial_data["total_assets"],
        errors="coerce",
    ).sum()
    aligned["asset_percentage"] = (
        aligned["total_assets"] / total_group_assets if total_group_assets > 0 else pd.NA
    )

    drop_columns = [
        "_entity_key",
        "entity_name_financial",
        "total_assets_financial",
    ]
    aligned = aligned.drop(
        columns=[column for column in drop_columns if column in aligned]
    )

    missing_group_entities = int(len(group_df) - matched_entities)
    return aligned, {
        "provided": True,
        "used": True,
        "path": financial_file,
        "matched_rows": matched_rows,
        "matched_entities": matched_entities,
        "ignored_rows": ignored_rows,
        "missing_group_entities": missing_group_entities,
        "message": (
            f"Financial data: provided and used from {financial_file}; "
            f"matched {matched_rows} financial row(s) across "
            f"{matched_entities} group entity/entities."
        ),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from modular.data.client_data_intake import load_client_data

    templates_dir = project_root / "templates"
    outputs_dir = project_root / "outputs_crewai"

    # Future intake sources may resolve from Azure Blob Storage / Azure SQL first.
    loaded_client_data = load_client_data(
        templates_dir / "group_entities_template.csv",
        templates_dir / "findings_template.csv",
        templates_dir / "financial_data_template.csv",
    )

    identified_risks = discover_risks(loaded_client_data)
    print(identified_risks)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    identified_risks.to_csv(outputs_dir / "identified_risks.csv", index=False)
