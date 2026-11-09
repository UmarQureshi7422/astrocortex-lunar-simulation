from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "dust_concentration",
    "electrostatic_charge",
    "surface_temperature",
    "vibration",
]
RISK_ORDER = ["LOW", "MODERATE", "HIGH"]

def create_synthetic_dataset(n=2400, seed=42):
    """Generate clearly labelled synthetic teaching data, not lunar measurements."""
    rng = np.random.default_rng(seed)
    dust = rng.uniform(0, 100, n)
    charge = rng.uniform(0, 100, n)
    temperature = rng.uniform(-150, 80, n)
    vibration = rng.uniform(0, 1, n)

    # Illustrative risk index used only to create a reproducible prototype dataset.
    # It is not a validated physical lunar-dust model.
    risk_index = (
        0.48 * dust
        + 0.34 * charge
        + 18.0 * vibration
        + 0.10 * np.abs(temperature + 35)
        + rng.normal(0, 8, n)
    )
    risk = np.where(
        risk_index < 43, "LOW",
        np.where(risk_index < 78, "MODERATE", "HIGH")
    )

    return pd.DataFrame({
        "dust_concentration": dust.round(3),
        "electrostatic_charge": charge.round(3),
        "surface_temperature": temperature.round(3),
        "vibration": vibration.round(4),
        "risk_index_illustrative": risk_index.round(3),
        "risk": risk,
    })

def main():
    df = create_synthetic_dataset()
    df.to_csv(DATA_DIR / "astroCortex_simulated_data.csv", index=False)

    X = df[FEATURES]
    y = df["risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=4, class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
            "Test Samples": len(y_test),
        })
        safe_name = {
            "Logistic Regression": "logistic_regression.pkl",
            "Decision Tree": "decision_tree.pkl",
            "Random Forest": "random_forest.pkl",
        }[name]
        joblib.dump(model, MODEL_DIR / safe_name)

    comparison = pd.DataFrame(rows).sort_values("F1", ascending=False)
    comparison.to_csv(DATA_DIR / "model_comparison.csv", index=False)

    print("AstroCortex training complete.")
    print(f"Dataset: {len(df)} synthetic rows")
    print("Class distribution:")
    print(df["risk"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")
    print("\nHeld-out test performance:")
    print(comparison.to_string(index=False, formatters={
        "Accuracy": "{:.1%}".format,
        "Precision": "{:.1%}".format,
        "Recall": "{:.1%}".format,
        "F1": "{:.1%}".format,
    }))
    print("\nSaved dataset, model files, and comparison table.")

if __name__ == "__main__":
    main()
