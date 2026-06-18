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
        for column in ["liquidity_ratio", "debt_ratio", "total_assets"]:
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

            total_assets = row.get("total_assets")
            if (
                total_group_assets > 0
                and pd.notna(total_assets)
                and total_assets / total_group_assets >= 0.15
            ):
                assets_percentage = total_assets / total_group_assets
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
            findings["severity"].astype(str).str.strip().str.casefold() == "high"
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
