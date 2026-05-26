"""Pydantic configuration schema models."""
from typing import Optional
from pydantic import BaseModel, Field


class ScopeRulesConfig(BaseModel):
    """Configuration for scope classification rules."""

    asset_percentage_threshold: float = Field(
        default=0.15,
        description="Asset percentage threshold for Full Scope classification",
        ge=0.0,
        le=1.0,
    )
    high_risk_scope: str = Field(
        default="Specific Procedures",
        description="Scope recommendation for high-risk entities",
    )
    default_scope: str = Field(
        default="Analytical Procedures",
        description="Default scope for low-risk entities below threshold",
    )
    full_scope_label: str = Field(
        default="Full Scope",
        description="Label for full scope audits",
    )


class CoverageConfig(BaseModel):
    """Configuration for coverage calculation."""

    minimum_coverage_percentage: float = Field(
        default=70.0,
        description="Minimum required audit coverage percentage",
        ge=0.0,
        le=100.0,
    )
    included_scopes: list[str] = Field(
        default=["Full Scope", "Specific Procedures"],
        description="Scopes to include in coverage calculation",
    )


class AuditConfig(BaseModel):
    """Configuration for audit instructions."""

    full_scope_template: str = Field(
        default=(
            "For {entity} in {country}, perform full scope audit procedures. "
            "The component should be audited comprehensively, including financial statements, "
            "key balances, revenue, expenses, assets, liabilities, and internal controls."
        ),
        description="Template for full scope audit instructions",
    )
    specific_procedures_template: str = Field(
        default=(
            "For {entity} in {country}, perform specific audit procedures focused on high-risk areas. "
            "Pay special attention to significant account balances, unusual transactions, "
            "management estimates, and areas with increased risk."
        ),
        description="Template for specific procedures audit instructions",
    )
    analytical_procedures_template: str = Field(
        default=(
            "For {entity} in {country}, perform analytical procedures only. "
            "Review financial trends, compare current results with prior periods, "
            "and investigate any unusual fluctuations."
        ),
        description="Template for analytical procedures audit instructions",
    )


class AppConfig(BaseModel):
    """Root application configuration."""

    scope_rules: ScopeRulesConfig = Field(default_factory=ScopeRulesConfig)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)

    # Data paths
    group_data_path: str = Field(
        default="data/group_structure.json",
        description="Path to group structure data",
    )
    findings_data_path: str = Field(
        default="data/findings_repo.csv",
        description="Path to findings repository",
    )

    # Output paths
    output_directory: str = Field(
        default="outputs",
        description="Output directory for results",
    )

    class Config:
        """Pydantic config."""
        extra = "forbid"
