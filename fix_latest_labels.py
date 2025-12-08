# fix_latest_labels.py

import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
PRICE_FILE = "price_history.csv"

def main():
    path = os.path.join(DATA_DIR, PRICE_FILE)
    df = pd.read_csv(path, parse_dates=["date"])

    # Sort to be safe
    df = df.sort_values(["product_id", "size", "date"]).reset_index(drop=True)

    # Get the latest row for each (product_id, size)
    latest = df.groupby(["product_id", "size"]).tail(1)

    latest_indices = latest.index.tolist()
    n_latest = len(latest_indices)

    if n_latest == 0:
        print("No rows found in price_history.csv")
        return

    # We'll force about 40% of latest rows to be good buys (label 1)
    np.random.seed(123)
    num_good = max(1, int(0.4 * n_latest))

    # Start by setting all latest rows to 0
    df.loc[latest_indices, "is_good_buy"] = 0

    # Randomly pick some indices among latest to set to 1
    chosen_indices = np.random.choice(latest_indices, size=num_good, replace=False)
    df.loc[chosen_indices, "is_good_buy"] = 1

    # Save back
    df.to_csv(path, index=False)
    print(f"Updated latest rows: set {num_good} of {n_latest} as is_good_buy = 1")

    # Optional sanity check
    latest_after = df.groupby(["product_id", "size"]).tail(1)
    print("Latest label distribution:")
    print(latest_after["is_good_buy"].value_counts())

if __name__ == "__main__":
    main()