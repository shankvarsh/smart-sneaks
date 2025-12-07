# train_buy_now_classifier.py

import json
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from features_and_data import build_features

MODELS_DIR = "models"

def main():
    # 1. Build features and target
    X, y_class, _ = build_features()

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )

    # 3. Define model
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    # 4. Train
    clf.fit(X_train, y_train)

    # 5. Predict
    y_pred = clf.predict(X_test)

    # 6. Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }

    # 7. Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 8. Save model
    model_path = os.path.join(MODELS_DIR, "buy_now_classifier.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved classifier model to {model_path}")

    # 9. Save metrics to JSON
    metrics_path = os.path.join(MODELS_DIR, "buy_now_classifier_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")

    # 10. Print key metrics in console
    print("\n=== Buy-Now Classifier Metrics ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")

if __name__ == "__main__":
    main()