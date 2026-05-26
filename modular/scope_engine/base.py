"""Abstract base class for scope recommendation engines."""
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from ..domain.models import ScopeRecommendation


class ScopeRecommendationEngine(ABC):
    """Abstract base class for scope recommendation engines."""

    @abstractmethod
    def recommend_scope(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """Generate scope recommendations for group entities.

        Args:
            group_df: DataFrame with entity data (Entity, Country, Assets, Risk_Level)

        Returns:
            DataFrame with original data plus Recommended_Scope column
        """
        pass

    @abstractmethod
    def get_recommendation(self, entity_data: dict) -> ScopeRecommendation:
        """Get scope recommendation for a single entity.

        Args:
            entity_data: Dictionary with entity information

        Returns:
            ScopeRecommendation object
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name for identification."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Engine version."""
        pass
