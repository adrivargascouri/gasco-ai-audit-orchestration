"""Model artifact paths and persistence helpers."""
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
SCOPE_MODEL_FILENAME = "scope_model.pkl"
SCOPE_MODEL_PATH = MODELS_DIR / SCOPE_MODEL_FILENAME

TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset.csv"
TARGET_COLUMN = "target_scope"
SCOPE_FEATURE_COLUMNS = [
    "revenue",
    "assets",
    "revenue_percentage",
    "assets_percentage",
    "risk_level_encoded",
    "country_risk_score",
    "prior_findings_count",
    "severe_findings_count",
    "growth_rate",
    "liquidity_ratio",
    "manual_risk_flag_encoded",
]


def ensure_models_dir() -> Path:
    """Ensure the local models directory exists."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def save_model_artifact(model: Any, model_path: Path = SCOPE_MODEL_PATH) -> Path:
    """Persist a model artifact and return its path."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_model_artifact(model_path: Path = SCOPE_MODEL_PATH) -> Any:
    """Load a persisted model artifact."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)
