
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# WLAN Sentinel AI — demonstration prototype
# IMPORTANT: This is synthetic data, not captured WLAN traffic.

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Generate illustrative NORMAL traffic records.
normal = pd.DataFrame({
    "duration": rng.uniform(2, 60, 600),
    "packet_count": rng.integers(20, 400, 600),
    "bytes_sent": rng.integers(500, 40000, 600),
    "label": "Normal",
})

# Generate illustrative SUSPICIOUS traffic records.
suspicious = pd.DataFrame({
    "duration": rng.uniform(1, 30, 400),
    "packet_count": rng.integers(200, 1500, 400),
    "bytes_sent": rng.integers(15000, 150000, 400),
    "label": "Suspicious",
})

# Combine and shuffle the synthetic examples.
data = pd.concat([normal, suspicious], ignore_index=True)
data = data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

features = ["duration", "packet_count", "bytes_sent"]
X = data[features]
y = data["label"]

# Evaluate on a held-out portion of the synthetic data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_SEED,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_SEED,
    class_weight="balanced",
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\n=== WLAN Sentinel AI: ML Prototype ===")
print("Training data: SYNTHETIC demonstration records only.")
print("This is not validated WLAN intrusion detection.")
print(f"Held-out synthetic test accuracy: {accuracy_score(y_test, predictions):.3f}")
print("\nClassification report on synthetic test records:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the synthetic dataset and trained model in the project folder.
project_folder = Path(__file__).resolve().parent
data.to_csv(project_folder / "demo_training_data.csv", index=False)

joblib.dump(
    {
        "model": model,
        "features": features,
        "data_type": "synthetic demonstration data",
        "validated_on_real_wlan": False,
    },
    project_folder / "wlan_demo_model.joblib",
)

print("\nSaved: demo_training_data.csv")
print("Saved: wlan_demo_model.joblib")
print("Next: test the generated model before connecting it to the dashboard.")