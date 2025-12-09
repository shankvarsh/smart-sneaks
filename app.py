# app.py

from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
from features_and_data import load_price_data
from smart_sneaks_engine import get_recommendation

app = Flask(__name__)

# Paths
DATA_DIR = "data"
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")

# Load data once at startup
df_products = pd.read_csv(PRODUCTS_CSV)
df_price = load_price_data()

def get_latest_price_snapshot():
    """
    Returns:
      latest_rows: latest row per (product_id, size)
      min_price_per_product: DataFrame with columns ['product_id', 'selling_price']
    """
    # latest row per product+size
    latest = df_price.sort_values("date").groupby(["product_id", "size"]).tail(1)

    # min price per product across sizes
    min_by_product = (
        latest.groupby("product_id")["selling_price"]
        .min()
        .reset_index()
        .rename(columns={"selling_price": "min_selling_price"})
    )

    return latest, min_by_product

@app.route("/")
def home():
    """
    Home page: show list of products with image, name, and current min price.
    """
    _, min_price_per_product = get_latest_price_snapshot()

    products_with_price = df_products.merge(
        min_price_per_product, on="product_id", how="left"
    )

    # Convert to list of dicts for template
    products_list = products_with_price.to_dict(orient="records")

    return render_template("index.html", products=products_list)

@app.route("/product/<product_id>")
def product_detail(product_id):
    """
    Product page: details for one product + size selector + charts + AI recommendation.
    """
    # Get product row
    product_df = df_products[df_products["product_id"] == product_id]
    if product_df.empty:
        return f"Product {product_id} not found", 404
    product = product_df.iloc[0].to_dict()

    # All rows for this product
    df_ps_all = df_price[df_price["product_id"] == product_id]
    if df_ps_all.empty:
        return f"No price data for product {product_id}", 404

    # All sizes available
    sizes = sorted(df_ps_all["size"].unique().tolist())

    # Selected size from query param, default first
    size_param = request.args.get("size")
    if size_param is None:
        selected_size = sizes[0]
    else:
        try:
            selected_size = float(size_param)
        except ValueError:
            selected_size = sizes[0]

    # Time series for this product + size
    df_ps_size = df_ps_all[df_ps_all["size"] == selected_size].sort_values("date")
    if df_ps_size.empty:
        return f"No price data for product {product_id} size {selected_size}", 404

    # History window for charts (last N days)
    hist_days = 60
    df_hist = df_ps_size.tail(hist_days)

    dates = df_hist["date"].dt.strftime("%Y-%m-%d").tolist()
    prices = df_hist["selling_price"].tolist()

    # --- Best buy date in 2026 ---
    df_2026 = df_ps_size[df_ps_size["date"].dt.year == 2026]
    if df_2026.empty:
        best_buy_date_2026 = None
        best_buy_price_2026 = None
    else:
        idx_min = df_2026["selling_price"].idxmin()
        best_row_2026 = df_2026.loc[idx_min]
        best_buy_date_2026 = best_row_2026["date"].date().isoformat()
        best_buy_price_2026 = float(best_row_2026["selling_price"])

    # --- AI recommendation from engine ---
    recommendation = get_recommendation(product_id, selected_size)

    current_price = recommendation["current_price"]
    mrp = recommendation["mrp"]
    current_discount_pct = recommendation["current_discount_pct"]
    future_drop_pct = recommendation["predicted_future_drop_pct"]

    # Estimated future best price (simple bar chart)
    estimated_best_price = round(current_price * (1 - future_drop_pct / 100.0), 2)
    future_date_label = "Next 30 days (est.)"

    # Latest snapshot row for this size
    latest_row = df_ps_size.iloc[-1].to_dict()

    # --- Out-of-stock: estimate restock ---
    restock_date = None
    restock_price = None
    latest_date = df_ps_size["date"].max()
    if latest_row["inventory_level"] == 0:
        future_rows = df_ps_size[df_ps_size["date"] > latest_date]
        future_instock = future_rows[future_rows["inventory_level"] > 0]
        if not future_instock.empty:
            first_restock = future_instock.iloc[0]
            restock_date = first_restock["date"].date().isoformat()
            restock_price = float(first_restock["selling_price"])

    # --- Series summary for explanation card ---
    if len(df_hist) >= 2:
        start_price = float(df_hist["selling_price"].iloc[0])
        end_price = float(df_hist["selling_price"].iloc[-1])
        min_price = float(df_hist["selling_price"].min())
        max_price = float(df_hist["selling_price"].max())
        min_date = df_hist.loc[df_hist["selling_price"].idxmin()]["date"].date().isoformat()
        max_date = df_hist.loc[df_hist["selling_price"].idxmax()]["date"].date().isoformat()
        avg_discount = float(df_hist["discount_pct"].mean())
        volatility = float(df_hist["selling_price"].std())

        if end_price < start_price * 0.97:
            trend = "a clear downward trend (prices have fallen over this period)"
        elif end_price > start_price * 1.03:
            trend = "a clear upward trend (prices have risen over this period)"
        else:
            trend = "a relatively flat pattern with small ups and downs"

        series_summary = (
            f"Over the last {len(df_hist)} days, this size started around ₹{start_price:.0f} "
            f"and is currently around ₹{end_price:.0f}, showing {trend}. "
            f"The lowest observed price in this window was ₹{min_price:.0f} on {min_date}, "
            f"while the highest reached ₹{max_price:.0f} on {max_date}. "
            f"Average discount during this period was about {avg_discount * 100:.1f}% "
            f"with a price volatility (standard deviation) of roughly ₹{volatility:.0f}."
        )
    else:
        series_summary = (
            "Not enough historical data is available to summarise the recent price behaviour for this size."
        )

    return render_template(
        "product_detail.html",
        product=product,
        sizes=sizes,
        selected_size=selected_size,
        dates=dates,
        prices=prices,
        recommendation=recommendation,
        estimated_best_price=estimated_best_price,
        future_date_label=future_date_label,
        latest_row=latest_row,
        mrp=mrp,
        current_price=current_price,
        current_discount_pct=current_discount_pct,
        best_buy_date_2026=best_buy_date_2026,
        best_buy_price_2026=best_buy_price_2026,
        restock_date=restock_date,
        restock_price=restock_price,
        series_summary=series_summary,
    )

@app.route("/analysis")
def data_analysis():
    """
    Data analysis / methodology overview page.
    Uses the actual synthetic dataset to compute stats that we then visualise with Chart.js,
    and shows notebook-like code snippets.
    """
    df = df_price  # global loaded via load_price_data()
    products = df_products

    total_products = products["product_id"].nunique()
    total_rows = len(df)
    date_min = df["date"].min().date().isoformat()
    date_max = df["date"].max().date().isoformat()

    # Distribution of discount_pct (bucketed)
    discount_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    discount_labels = ["0–10%", "10–20%", "20–30%", "30–40%", "40–50%", "50%+"]

    cut_discount = pd.cut(
        df["discount_pct"],
        bins=discount_bins,
        labels=discount_labels,
        include_lowest=True,
        right=False,
    )
    discount_counts = (
        cut_discount.value_counts().reindex(discount_labels).fillna(0).astype(int).tolist()
    )

    # Distribution of inventory levels (bucketed)
    inv_bins = [0, 1, 10, 20, 50, 100, 1000]
    inv_labels = ["0", "1–9", "10–19", "20–49", "50–99", "100+"]

    cut_inv = pd.cut(
        df["inventory_level"],
        bins=inv_bins,
        labels=inv_labels,
        include_lowest=True,
        right=False,
    )
    inv_counts = (
        cut_inv.value_counts().reindex(inv_labels).fillna(0).astype(int).tolist()
    )

    # Label distribution for is_good_buy
    label_counts_series = df["is_good_buy"].value_counts()
    good_count = int(label_counts_series.get(1, 0))
    bad_count = int(label_counts_series.get(0, 0))

    # Average future_drop_pct for good vs bad labels
    grouped_drop = df.groupby("is_good_buy")["future_drop_pct"].mean()
    avg_drop_good = float(grouped_drop.get(1, 0.0))
    avg_drop_bad = float(grouped_drop.get(0, 0.0))

    # Scenario-wise average discount
    if "scenario" in df.columns:
        scenario_stats = (
            df.groupby("scenario")["discount_pct"]
            .mean()
            .sort_values(ascending=False)
        )
        scenario_names = scenario_stats.index.tolist()
        scenario_avg_discount = (scenario_stats.values * 100).round(1).tolist()
    else:
        scenario_names = []
        scenario_avg_discount = []

    # Notebook-like code snippets to show on the page
    code_read_data = """\
import pandas as pd

df_price = pd.read_csv("data/price_history.csv", parse_dates=["date"])
df_products = pd.read_csv("data/products.csv")

df_price.head()
"""
    code_features_targets = """\
from features_and_data import build_features

X, y_class, y_reg = build_features()

X.sample(5)      # feature matrix
y_class.head()   # is_good_buy labels
y_reg.head()     # future_drop_pct regression targets
"""
    code_training = """\
from train_buy_now_classifier import main as train_classifier
from train_future_drop_regressor import main as train_regressor

# Re-train models on the latest synthetic dataset
train_classifier()
train_regressor()
"""
    code_inference = """\
from smart_sneaks_engine import get_recommendation

rec = get_recommendation("NIK001", 9)
rec
"""

    return render_template(
        "analysis.html",
        total_products=total_products,
        total_rows=total_rows,
        date_min=date_min,
        date_max=date_max,
        discount_labels=discount_labels,
        discount_counts=discount_counts,
        inv_labels=inv_labels,
        inv_counts=inv_counts,
        good_count=good_count,
        bad_count=bad_count,
        avg_drop_good=avg_drop_good,
        avg_drop_bad=avg_drop_bad,
        scenario_names=scenario_names,
        scenario_avg_discount=scenario_avg_discount,
        code_read_data=code_read_data,
        code_features_targets=code_features_targets,
        code_training=code_training,
        code_inference=code_inference,
    )

@app.route("/admin/metrics")
def admin_metrics():
    """
    Hidden admin route: show model evaluation metrics.
    """
    clf_metrics_path = os.path.join("models", "buy_now_classifier_metrics.json")
    reg_metrics_path = os.path.join("models", "future_drop_regressor_metrics.json")

    import json

    with open(clf_metrics_path, "r") as f:
        clf_metrics = json.load(f)

    with open(reg_metrics_path, "r") as f:
        reg_metrics = json.load(f)

    return render_template(
        "admin_metrics.html",
        clf=clf_metrics,
        reg=reg_metrics,
    )
    
if __name__ == "__main__":
    app.run(debug=True)