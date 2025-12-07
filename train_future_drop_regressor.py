# train_future_drop_regressor.py

import json
import os

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from features_and_data import build_features

MODELS_DIR = "models"

def main():
    # 1. Build features and targets
    X, _, y_reg = build_features()

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_reg,
        test_size=0.2,
        random_state=42
    )

    # 3. Define model
    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    # 4. Train
    reg.fit(X_train, y_train)

    # 5. Predict
    y_pred = reg.predict(X_test)

    # 6. Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

    # 7. Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 8. Save model
    model_path = os.path.join(MODELS_DIR, "future_drop_regressor.joblib")
    joblib.dump(reg, model_path)
    print(f"Saved regressor model to {model_path}")

    # 9. Save metrics to JSON
    metrics_path = os.path.join(MODELS_DIR, "future_drop_regressor_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")

    # 10. Print metrics
    print("\n=== Future Drop Regressor Metrics ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

if __name__ == "__main__":
    main()