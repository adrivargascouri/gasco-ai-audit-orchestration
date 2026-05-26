import numpy as np
from typing import Dict, Any, List


class RiskLevelEncoder:
    """Encode risk_level categorical variable to numeric values."""

    MAPPING = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Critical": 4,
    }

    REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}

    @classmethod
    def encode(cls, value: str) -> int:
        if value not in cls.MAPPING:
            raise ValueError(f"Unknown risk level: {value}")
        return cls.MAPPING[value]

    @classmethod
    def encode_array(cls, values: List[str]) -> np.ndarray:
        return np.array([cls.encode(v) for v in values])

    @classmethod
    def decode(cls, value: int) -> str:
        if value not in cls.REVERSE_MAPPING:
            raise ValueError(f"Unknown encoded risk level: {value}")
        return cls.REVERSE_MAPPING[value]


class ManualRiskFlagEncoder:
    """Encode manual_risk_flag Yes/No to numeric values."""

    MAPPING = {"No": 0, "Yes": 1}
    REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}

    @classmethod
    def encode(cls, value: str) -> int:
        if value not in cls.MAPPING:
            raise ValueError(f"Unknown manual risk flag: {value}")
        return cls.MAPPING[value]

    @classmethod
    def encode_array(cls, values: List[str]) -> np.ndarray:
        return np.array([cls.encode(v) for v in values])

    @classmethod
    def decode(cls, value: int) -> str:
        if value not in cls.REVERSE_MAPPING:
            raise ValueError(f"Unknown encoded manual risk flag: {value}")
        return cls.REVERSE_MAPPING[value]


class TargetScopeEncoder:
    """Encode target_scope labels to numeric values."""

    MAPPING = {
        "Analytical Procedures": 0,
        "Specific Procedures": 1,
        "Full Scope": 2,
    }

    REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}

    @classmethod
    def encode(cls, value: str) -> int:
        if value not in cls.MAPPING:
            raise ValueError(f"Unknown target scope: {value}")
        return cls.MAPPING[value]

    @classmethod
    def encode_array(cls, values: List[str]) -> np.ndarray:
        return np.array([cls.encode(v) for v in values])

    @classmethod
    def decode(cls, value: int) -> str:
        if value not in cls.REVERSE_MAPPING:
            raise ValueError(f"Unknown encoded target scope: {value}")
        return cls.REVERSE_MAPPING[value]
