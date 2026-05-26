import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from .encoders import RiskLevelEncoder, ManualRiskFlagEncoder, TargetScopeEncoder


class FeatureEngineer:
    """
    Load historical audit data, engineer features, and prepare training dataset.
    """

    REQUIRED_COLUMNS = {
        "entity",
        "country",
        "revenue",
        "assets",
        "revenue_percentage",
        "assets_percentage",
        "risk_level",
        "country_risk_score",
        "prior_findings_count",
        "severe_findings_count",
        "growth_rate",
        "liquidity_ratio",
        "manual_risk_flag",
        "target_scope",
    }

    ML_FEATURES = [
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

    def __init__(
        self,
        raw_data_path: str = "data/raw/historical_audit_data.csv",
        processed_data_path: str = "data/processed/training_dataset.csv",
    ):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.df = None
        self.df_processed = None

    def load_data(self) -> pd.DataFrame:
        """Load historical audit data."""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data not found: {self.raw_data_path}")
        self.df = pd.read_csv(self.raw_data_path)
        return self.df

    def validate_columns(self) -> bool:
        """Validate required columns exist."""
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True

    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values."""
        self.df = self.df.dropna()
        return self.df

    def encode_categorical_variables(self) -> pd.DataFrame:
        """Encode categorical variables."""
        self.df["risk_level_encoded"] = self.df["risk_level"].apply(
            RiskLevelEncoder.encode
        )
        self.df["manual_risk_flag_encoded"] = self.df["manual_risk_flag"].apply(
            ManualRiskFlagEncoder.encode
        )
        self.df["target_scope_encoded"] = self.df["target_scope"].apply(
            TargetScopeEncoder.encode
        )
        return self.df

    def select_ml_features(self) -> pd.DataFrame:
        """Select features for ML model."""
        feature_cols = self.ML_FEATURES + ["target_scope_encoded"]
        self.df_processed = self.df[feature_cols].copy()
        self.df_processed.columns = self.ML_FEATURES + ["target_scope"]
        return self.df_processed

    def export_training_dataset(self) -> Path:
        """Export processed dataset to CSV."""
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.df_processed.to_csv(self.processed_data_path, index=False)
        return self.processed_data_path

    def process(self) -> pd.DataFrame:
        """Execute full feature engineering pipeline."""
        self.load_data()
        self.validate_columns()
        self.handle_missing_values()
        self.encode_categorical_variables()
        self.select_ml_features()
        self.export_training_dataset()
        return self.df_processed

    def get_statistics(self) -> dict:
        """Get statistics about processed data."""
        return {
            "rows": len(self.df_processed),
            "columns": len(self.df_processed.columns),
            "missing_values": self.df_processed.isnull().sum().sum(),
            "target_scope_distribution": self.df_processed["target_scope"].value_counts().to_dict(),
        }


def main():
    """Run feature engineering pipeline."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / "data/raw/historical_audit_data.csv"
    processed_data_path = project_root / "data/processed/training_dataset.csv"

    engineer = FeatureEngineer(
        raw_data_path=str(raw_data_path),
        processed_data_path=str(processed_data_path),
    )

    print("Starting feature engineering pipeline...")
    try:
        df_processed = engineer.process()
        stats = engineer.get_statistics()

        print(f"\n[OK] Feature engineering completed successfully!")
        print(f"\nDataset Statistics:")
        print(f"  Rows: {stats['rows']}")
        print(f"  Columns: {stats['columns']}")
        print(f"  Missing values: {stats['missing_values']}")
        print(f"\nTarget Scope Distribution:")
        for scope, count in sorted(stats["target_scope_distribution"].items()):
            print(f"  {scope}: {count}")

        print(f"\nProcessed dataset saved to: {processed_data_path}")
        print(f"\nFirst 5 rows:")
        print(df_processed.head())

    except Exception as e:
        print(f"[ERROR] Error during feature engineering: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
