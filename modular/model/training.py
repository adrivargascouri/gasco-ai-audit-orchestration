"""Train the local supervised scope recommendation model."""
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from modular.features.encoders import TargetScopeEncoder
from modular.model.model_registry import (
    SCOPE_FEATURE_COLUMNS,
    SCOPE_MODEL_PATH,
    TARGET_COLUMN,
    TRAINING_DATA_PATH,
    save_model_artifact,
)


TEST_SIZE = 0.25
RANDOM_STATE = 42


def load_training_dataset(data_path: Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    """Load and validate the processed training dataset."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = set(SCOPE_FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Training dataset missing columns: {sorted(missing_columns)}")

    missing_values = df[SCOPE_FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().sum()
    if missing_values:
        raise ValueError(f"Training dataset contains {missing_values} missing values")

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split processed data into model features and target."""
    return df[SCOPE_FEATURE_COLUMNS], df[TARGET_COLUMN]


def build_classifier() -> RandomForestClassifier:
    """Create the Phase 2B baseline classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


def train_scope_model(
    data_path: Path = TRAINING_DATA_PATH,
    model_path: Path = SCOPE_MODEL_PATH,
) -> tuple[RandomForestClassifier, float, str, Path]:
    """Train, evaluate, and save the scope model."""
    df = load_training_dataset(data_path)
    x, y = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_classifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    labels = sorted(y.unique())
    target_names = [TargetScopeEncoder.decode(int(label)) for label in labels]
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    model.feature_columns_ = SCOPE_FEATURE_COLUMNS
    model.target_column_ = TARGET_COLUMN
    model.target_labels_ = {
        int(label): TargetScopeEncoder.decode(int(label)) for label in labels
    }
    model.training_accuracy_ = float(accuracy)
    model.classification_report_ = report

    saved_path = save_model_artifact(model, model_path)

    print("GASCO Phase 2B Scope Model Training")
    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print(f"Model saved to: {saved_path}")

    return model, float(accuracy), report, saved_path


def main() -> None:
    """Run model training from the command line."""
    train_scope_model()


if __name__ == "__main__":
    main()
