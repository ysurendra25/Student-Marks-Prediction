"""
Student Marks Prediction
========================
A simple machine learning project that predicts a student's final marks
(0-100) from three input features:

    - study_hours       : average study hours per day
    - attendance_percent: class attendance (%)
    - previous_score    : score in the previous exam (0-100)

Model: Linear Regression (scikit-learn)

Usage:
    python student_marks_prediction.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = os.path.join("data", "student_data.csv")
PLOT_PATH = os.path.join("plots", "actual_vs_predicted.png")
RANDOM_STATE = 42


def create_dataset(n_samples: int = 120, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a small, reproducible synthetic dataset.

    Marks are generated from a simple relationship so that the linear model
    has a clear, learnable signal:

        marks = 5.0 * study_hours + 0.30 * attendance + 0.55 * previous_score + noise
    """
    rng = np.random.default_rng(seed)

    study_hours = np.round(rng.uniform(0.5, 9.0, n_samples), 1)
    attendance_percent = np.round(rng.uniform(55, 100, n_samples), 1)
    previous_score = np.round(rng.uniform(35, 98, n_samples), 1)

    noise = rng.normal(0, 2.5, n_samples)
    marks = (
        5.0 * study_hours
        + 0.30 * attendance_percent
        + 0.55 * previous_score
        + noise
    )
    marks = np.clip(np.round(marks, 1), 0, 100)

    return pd.DataFrame(
        {
            "study_hours": study_hours,
            "attendance_percent": attendance_percent,
            "previous_score": previous_score,
            "marks": marks,
        }
    )


def load_data() -> pd.DataFrame:
    """Load the dataset, creating it first if it does not exist."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df = create_dataset()
        df.to_csv(DATA_PATH, index=False)
        print(f"Created dataset at {DATA_PATH} ({len(df)} rows)")
    return pd.read_csv(DATA_PATH)


def main() -> None:
    df = load_data()
    print("\nDataset preview:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe().round(2))

    features = ["study_hours", "attendance_percent", "previous_score"]
    X = df[features]
    y = df["marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\nModel: Linear Regression")
    print(f"  MAE  : {mae:.2f} marks")
    print(f"  RMSE : {rmse:.2f} marks")
    print(f"  R^2  : {r2:.3f}")

    print("\nLearned coefficients:")
    for name, coef in zip(features, model.coef_):
        print(f"  {name:20s}: {coef:+.3f}")
    print(f"  {'intercept':20s}: {model.intercept_:+.3f}")

    # Example prediction for a new student
    sample = pd.DataFrame(
        [{"study_hours": 6.0, "attendance_percent": 85.0, "previous_score": 72.0}]
    )
    predicted = model.predict(sample)[0]
    print("\nExample prediction")
    print("  Input : 6.0 study hours/day, 85% attendance, 72 previous score")
    print(f"  Output: {predicted:.1f} marks (predicted)")

    # Save an actual-vs-predicted plot
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=1.5)
    plt.xlabel("Actual marks")
    plt.ylabel("Predicted marks")
    plt.title("Student Marks Prediction - Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    plt.close()
    print(f"\nSaved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
