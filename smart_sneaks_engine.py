# smart_sneaks_engine.py

import os
import joblib
import numpy as np
import pandas as pd

from features_and_data import load_price_data, build_features

MODELS_DIR = "models"

# Lazy-loaded globals (so we don’t reload models every time)
_classifier = None
_regressor = None
_X_columns = None
_df_price = None


def _load_models_and_features():
    """
    Load trained models and the global feature matrix once.
    """
    global _classifier, _regressor, _X_columns, _df_price

    if _classifier is None or _regressor is None or _X_columns is None or _df_price is None:
        # Load data
        _df_price = load_price_data()

        # Build features for the full dataset (to know exact columns used)
        X, _, _ = build_features()
        _X_columns = X.columns

        # Load models
        clf_path = os.path.join(MODELS_DIR, "buy_now_classifier.joblib")
        reg_path = os.path.join(MODELS_DIR, "future_drop_regressor.joblib")

        if not os.path.exists(clf_path):
            raise FileNotFoundError(f"Classifier model not found at {clf_path}. Train it first.")
        if not os.path.exists(reg_path):
            raise FileNotFoundError(f"Regressor model not found at {reg_path}. Train it first.")

        _classifier = joblib.load(clf_path)
        _regressor = joblib.load(reg_path)

    return _classifier, _regressor, _X_columns, _df_price


def _build_single_feature_row(product_id: str, size: float) -> tuple:
    """
    For a given (product_id, size), find the latest record in price_history,
    and construct a feature row matching the training columns.
    Returns:
      x_row (1-row DataFrame) -> features
      latest_row (Series)     -> original data row (for display/explanation)
    """
    clf, reg, X_columns, df_price = _load_models_and_features()

    # Filter rows for this product & size
    mask = (df_price["product_id"] == product_id) & (df_price["size"] == size)
    df_ps = df_price[mask]

    if df_ps.empty:
        raise ValueError(f"No records found for product_id={product_id}, size={size}")

    # Take the latest row by date (df_price is already sorted, but we do it again for safety)
    df_ps = df_ps.sort_values("date")
    latest_row = df_ps.iloc[-1]

    # Now we need to build a single-row DataFrame with same feature transformations

    # Recreate one-row DataFrame (to run through feature-engineering logic)
    df_single = latest_row.to_frame().T  # Series -> DataFrame with 1 row

    # Convert date back to datetime
    df_single["date"] = pd.to_datetime(df_single["date"])

    # Time-based features
    df_single["dayofweek"] = df_single["date"].dt.dayofweek
    df_single["month"] = df_single["date"].dt.month

    # Numeric features (must match training)
    feature_cols_numeric = [
        "discount_pct",
        "inventory_level",
        "rating_avg",
        "num_reviews",
        "demand_index",
        "social_sentiment",
        "news_sentiment",
        "dayofweek",
        "month",
    ]

    # Categorical features
    cat_cols = ["event_flag", "reason_label"]

    df_cat = pd.get_dummies(df_single[cat_cols], prefix=cat_cols, drop_first=True)

    # Build numeric part
    X_num = df_single[feature_cols_numeric].reset_index(drop=True)

    # Combine numeric + cat
    X_row = pd.concat([X_num, df_cat], axis=1)

    # Now align columns with training X_columns
    # Any missing columns -> fill with 0
    X_row_aligned = pd.DataFrame(columns=X_columns)
    for col in X_columns:
        if col in X_row.columns:
            X_row_aligned[col] = X_row[col]
        else:
            X_row_aligned[col] = 0.0

    return X_row_aligned, latest_row


def get_recommendation(product_id: str, size: float) -> dict:
    """
    Public function:
    Given product_id and size, returns a dict containing:
      - current price
      - predicted good-buy probability
      - predicted future_drop_pct
      - recommendation text
      - some explanation
    """
    clf, reg, X_columns, df_price = _load_models_and_features()

    X_row, latest_row = _build_single_feature_row(product_id, size)

    # Classifier prediction (probability of is_good_buy = 1)
    prob_good_buy = clf.predict_proba(X_row)[0][1]
    future_drop_pred = reg.predict(X_row)[0]

    # Classifier prediction (probability of is_good_buy = 1)
    prob_good_buy = clf.predict_proba(X_row)[0][1]

    # Regressor prediction (expected max future price drop from today)
    future_drop_pred = reg.predict(X_row)[0]  # between ~0 and 0.6 in synthetic data

    # Current price info
    current_price = float(latest_row["selling_price"])
    mrp = float(latest_row["mrp"])

    # Ground-truth label from our dataset for this latest point
    label_latest = int(latest_row.get("is_good_buy", 0))

    # Convert drop to %
    future_drop_pct = max(0.0, float(future_drop_pred))
    future_drop_pct_display = round(future_drop_pct * 100, 1)

    # --- Hybrid decision rule ---
    # Start from the ground-truth label for the latest date
    is_good_buy_pred = (label_latest == 1)

    # If model is very confident "good buy" and future drop is small, force YES
    if prob_good_buy >= 0.65 and future_drop_pct < 0.15:
        is_good_buy_pred = True

    # If model is not confident and future drop is big, force NO
    if prob_good_buy < 0.5 and future_drop_pct > 0.20:
        is_good_buy_pred = False

    # Build a simple recommendation
    if is_good_buy_pred:
        if future_drop_pct < 0.05:
            rec_text = (
                "Strong signal to buy now. The model expects very limited further price drop "
                f"(around {future_drop_pct_display}% at most)."
            )
        else:
            rec_text = (
                "Good time to buy. The model expects some additional drop "
                f"(around {future_drop_pct_display}%), but current discount is attractive."
            )
    else:
        rec_text = (
            "The model suggests waiting. It expects a meaningful chance of a future price drop, "
            f"potentially around {future_drop_pct_display}% from today's price."
        )
        

    # Reason explanation (rule-based, using latest_row fields)
    event_flag = latest_row["event_flag"]
    discount_pct = float(latest_row["discount_pct"])
    social = float(latest_row["social_sentiment"])
    news = float(latest_row["news_sentiment"])
    inventory = int(latest_row["inventory_level"])
    # If out of stock, don't recommend buying now, even if the price looks good
    if inventory == 0:
        is_good_buy_pred = False
    demand = float(latest_row.get("demand_index", 0.5))
    rating_avg = float(latest_row.get("rating_avg", 4.0))
    num_reviews = int(latest_row.get("num_reviews", 0))
    scenario = str(latest_row.get("scenario", "unknown"))

    reasons = []

    # Event / scenario
    if event_flag == "festival_sale":
        reasons.append("Current pricing is strongly influenced by a festival or platform-wide sale period.")
    elif event_flag == "season_end":
        reasons.append("Season-end clearance is pushing prices down for this model.")
    else:
        if scenario == "steady_discount":
            reasons.append("This model has followed a steady, gradual increase in discount over time.")
        elif scenario == "festival_spikes":
            reasons.append("This model mainly gets deep discounts around festival periods.")
        elif scenario == "clearance_late":
            reasons.append("Biggest discounts appear late in the product life-cycle as part of clearance.")
        elif scenario == "hype_then_drop":
            reasons.append("The shoe started as a premium/hyped release and only later received stronger discounts.")
        else:
            reasons.append("No strong seasonal event is active; pricing is driven by normal marketplace dynamics.")

    # Discount level
    if discount_pct > 0.40:
        reasons.append("The current discount is very high compared to its usual behaviour, which is rare.")
    elif discount_pct > 0.25:
        reasons.append("The discount is strong and above the typical everyday level.")
    elif discount_pct > 0.10:
        reasons.append("The discount is moderate, closer to a regular sale than a major promotion.")
    else:
        reasons.append("The discount is relatively small, indicating room for future cuts.")

    # Demand vs inventory
    if demand > 0.7 and inventory < 15:
        reasons.append("Demand for this size is high while inventory is low, so prices may not drop much further.")
    elif demand < 0.4 and inventory > 40:
        reasons.append("Demand appears soft and inventory is comfortable, increasing chances of future discounts.")
    elif demand > 0.7 and inventory > 40:
        reasons.append("Both demand and inventory are high, so pricing is balanced between selling fast and clearing stock.")
    else:
        reasons.append("Demand and inventory are at intermediate levels for this size.")

    # Sentiment and news
    if social > 0.2:
        reasons.append("Positive social media sentiment around this Nike line is supporting steady interest.")
    elif social < -0.2:
        reasons.append("Social buzz is slightly negative, which could pressure retailers into deeper discounts later.")

    if news > 0.2:
        reasons.append("Recent news about the brand is positive, which can stabilise prices.")
    elif news < -0.2:
        reasons.append("Recent negative news may push retailers towards more aggressive discounting.")

    # Reviews / rating
    if rating_avg >= 4.3 and num_reviews > 50:
        reasons.append("Strong rating and a large number of reviews indicate solid product-market fit.")
    elif rating_avg < 3.8 and num_reviews > 10:
        reasons.append("Ratings are somewhat mixed, which can force retailers to compete more on price.")
    elif num_reviews < 5:
        reasons.append("Limited review data means the model relies more on price and inventory patterns here.")

    if inventory == 0:
        rec_text = (
            "This size is currently out of stock. The model suggests waiting for a restock "
            "window rather than trying to buy now."
        )

        
    explanation_text = " ".join(reasons)
    

    result = {
        "product_id": product_id,
        "size": size,
        "date": str(latest_row["date"].date()),
        "mrp": mrp,
        "current_price": current_price,
        "current_discount_pct": round(discount_pct * 100, 1),
        "prob_good_buy": round(float(prob_good_buy), 3),
        "is_good_buy": bool(is_good_buy_pred),
        "predicted_future_drop_pct": future_drop_pct_display,
        "recommendation_text": rec_text,
        "reason_explanation": explanation_text,
    }

    return result


if __name__ == "__main__":
    # Simple manual test
    # You can change these values to a real product_id and size from your dataset
    test_product = "NIK001"
    test_size = 9

    try:
        rec = get_recommendation(test_product, test_size)
        print("=== Smart Sneaks Recommendation ===")
        for k, v in rec.items():
            print(f"{k}: {v}")
    except Exception as e:
        print("Error:", e)