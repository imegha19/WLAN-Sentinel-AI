
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "synthetic_awid3_style_demo.csv"
MODEL_PATH = BASE_DIR / "wlan_multiclass_synthetic_model.joblib"
ENCODER_PATH = BASE_DIR / "wlan_multiclass_label_encoder.joblib"

TARGET_COLUMN = "label"
RANDOM_STATE = 42


def main():
    print("Loading synthetic WLAN dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Place synthetic_awid3_style_demo.csv in the project folder."
        )

    df = pd.read_csv(DATASET_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError("The dataset must contain a 'label' column.")

    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Use numeric columns as model inputs.
    features = df.drop(columns=[TARGET_COLUMN])
    features = features.select_dtypes(include=["number"])

    if features.empty:
        raise ValueError("No numeric feature columns were found.")

    if features.isna().any().any():
        raise ValueError(
            "Numeric features contain missing values. "
            "Inspect and clean the dataset before training."
        )

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df[TARGET_COLUMN].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Dataset rows: {len(df):,}")
    print(f"Numeric features: {len(features.columns)}")
    print(f"Classes: {list(encoder.classes_)}")
    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n=== SYNTHETIC DATASET EVALUATION ===")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(encoder.classes_))),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(features.columns),
            "target_column": TARGET_COLUMN,
            "dataset_type": "synthetic_simulated_wlan_traffic",
        },
        MODEL_PATH,
    )
    joblib.dump(encoder, ENCODER_PATH)

    print(f"\nModel saved to: {MODEL_PATH.name}")
    print(f"Label encoder saved to: {ENCODER_PATH.name}")
    print(
        "\nIMPORTANT: These results use synthetic data only. "
        "They do not establish real-world WLAN detection performance."
    )


if __name__ == "__main__":
    main()
