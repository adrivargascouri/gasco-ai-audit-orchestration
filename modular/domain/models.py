"""Domain entities for GASCO audit orchestrator."""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level enumeration."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ScopeType(str, Enum):
    """Audit scope type enumeration."""

    FULL_SCOPE = "Full Scope"
    SPECIFIC_PROCEDURES = "Specific Procedures"
    ANALYTICAL_PROCEDURES = "Analytical Procedures"


class Entity(BaseModel):
    """Entity (subsidiary) in the group."""

    entity: str = Field(..., description="Entity name")
    country: str = Field(..., description="Country of operation")
    revenue: float = Field(..., ge=0, description="Revenue amount")
    assets: float = Field(..., ge=0, description="Total assets")
    risk_level: RiskLevel = Field(..., description="Risk level classification")

    class Config:
        """Pydantic config."""
        from_attributes = True


class Finding(BaseModel):
    """Audit finding from findings repository."""

    entity: str = Field(..., description="Entity with finding")
    finding: str = Field(..., description="Description of finding")
    severity: RiskLevel = Field(..., description="Severity level")

    class Config:
        """Pydantic config."""
        from_attributes = True


class ScopeRecommendation(BaseModel):
    """Scope recommendation for an entity."""

    entity: str = Field(..., description="Entity name")
    country: str = Field(..., description="Country")
    assets: float = Field(..., ge=0, description="Total assets")
    asset_percentage: float = Field(
        ..., ge=0, le=100, description="Asset percentage of total group assets"
    )
    risk_level: RiskLevel = Field(..., description="Risk level")
    recommended_scope: ScopeType = Field(..., description="Recommended audit scope")

    class Config:
        """Pydantic config."""
        from_attributes = True


class CoverageSummary(BaseModel):
    """Summary of group audit coverage."""

    total_assets: float = Field(..., ge=0, description="Total group assets")
    covered_assets: float = Field(..., ge=0, description="Assets covered by audit")
    coverage_percentage: float = Field(
        ..., ge=0, le=100, description="Coverage percentage"
    )
    status: str = Field(..., description="Coverage status message")

    class Config:
        """Pydantic config."""
        from_attributes = True


class AuditInstruction(BaseModel):
    """Audit instruction for an entity."""

    entity: str = Field(..., description="Entity name")
    country: str = Field(..., description="Country")
    risk_level: RiskLevel = Field(..., description="Risk level")
    recommended_scope: ScopeType = Field(..., description="Recommended scope")
    audit_instruction: str = Field(..., description="Detailed audit instruction")

    class Config:
        """Pydantic config."""
        from_attributes = True
