# app.py

from flask import Flask, render_template, request
import os
import pandas as pd

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
    product = df_products[df_products["product_id"] == product_id]
    if product.empty:
        return f"Product {product_id} not found", 404
    product = product.iloc[0].to_dict()

    # All sizes available for this product
    df_ps_all = df_price[df_price["product_id"] == product_id]
    if df_ps_all.empty:
        return f"No price data for product {product_id}", 404

    sizes = sorted(df_ps_all["size"].unique().tolist())

    # Get selected size from query param, default to first size
    size_param = request.args.get("size")
    if size_param is None:
        selected_size = sizes[0]
    else:
        try:
            selected_size = float(size_param)
        except ValueError:
            selected_size = sizes[0]

    # Historical price for this product+size (past N days)
    df_ps_size = df_ps_all[df_ps_all["size"] == selected_size].sort_values("date")
    hist_days = 60
    df_hist = df_ps_size.tail(hist_days)

    dates = df_hist["date"].dt.strftime("%Y-%m-%d").tolist()
    prices = df_hist["selling_price"].tolist()

    # AI recommendation using our engine
    recommendation = get_recommendation(product_id, selected_size)

    current_price = recommendation["current_price"]
    mrp = recommendation["mrp"]
    current_discount_pct = recommendation["current_discount_pct"]
    future_drop_pct = recommendation["predicted_future_drop_pct"]

    estimated_best_price = round(current_price * (1 - future_drop_pct / 100.0), 2)
    future_date_label = "Next 30 days (est.)"

    latest_row = df_ps_size.iloc[-1].to_dict()

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