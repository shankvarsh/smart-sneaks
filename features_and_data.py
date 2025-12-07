import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
PRICE_FILE = "price_history.csv"

def load_price_data():
    """
    Loads the price history dataset from the data folder
    and sorts it by product_id, size, date so that
    indices are consistent for features and targets.
    """
    path = os.path.join(DATA_DIR, PRICE_FILE)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["product_id", "size", "date"]).reset_index(drop=True)
    return df

def build_features():
    """
    Loads data and returns:
      X          -> feature dataframe
      y_class    -> classification target (is_good_buy)
      y_reg      -> regression target (future_drop_pct)
    """
    df = load_price_data()

    # Basic time-based features
    df["dayofweek"] = df["date"].dt.dayofweek   # 0=Mon
    df["month"] = df["date"].dt.month

    # Select raw feature columns
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

    # Categorical columns (we will one-hot encode these)
    cat_cols = ["event_flag", "reason_label"]   # we can use reason_label as input too

    # One-hot encode categorical columns
    df_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=True)

    X = pd.concat([df[feature_cols_numeric].reset_index(drop=True),
                   df_cat.reset_index(drop=True)], axis=1)

    # Targets
    y_class = df["is_good_buy"].astype(int)
    y_reg = df["future_drop_pct"].astype(float)

    return X, y_class, y_reg