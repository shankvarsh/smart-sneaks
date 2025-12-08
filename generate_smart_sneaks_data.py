# generate_smart_sneaks_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

# -------------------------
# 1. Config
# -------------------------
N_PRODUCTS = 20
SIZES = [6, 7, 8, 9, 10, 11]

# 2 years of data: 2025 + 2026
START_DATE = datetime(2025, 1, 1)
DAYS = 730

DATE_RANGE = [START_DATE + timedelta(days=i) for i in range(DAYS)]

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Some fake big sale/festival dates (2025 & 2026)
festival_dates = [
    datetime(2025, 1, 26),   # Republic Day sale
    datetime(2025, 8, 15),   # Independence Day sale
    datetime(2025, 11, 1),   # Diwali-ish
    datetime(2026, 1, 26),
    datetime(2026, 8, 15),
    datetime(2026, 11, 1),
]

def close_to_festival(d, window=5):
    for fd in festival_dates:
        if abs((d - fd).days) <= window:
            return True
    return False

# -------------------------
# 2. Create products.csv
# -------------------------

nike_models = [
    "Nike Air Zoom Pegasus 39",
    "Nike Revolution 7",
    "Nike Air Max 270",
    "Nike Court Vision Low",
    "Nike Renew Run 4",
    "Nike Quest 5",
    "Nike Downshifter 12",
    "Nike Blazer Mid '77",
    "Nike Air Force 1 '07",
    "Nike Infinity Run Flyknit",
    "Nike React Miler",
    "Nike ZoomX Invincible",
    "Nike Air Zoom Structure",
    "Nike Metcon 8",
    "Nike Zoom Freak",
    "Nike Phantom GX Academy",
    "Nike G.T. Cut",
    "Nike Vaporfly",
    "Nike Air Zoom Alphafly",
    "Nike Zoom Fly 5"
]

categories = ["running", "casual", "basketball", "training"]
genders = ["men", "women", "unisex"]
colors = ["black", "white", "blue", "red", "grey", "multicolor"]

products = []
for i in range(N_PRODUCTS):
    pid = f"NIK{i+1:03d}"
    name = nike_models[i % len(nike_models)]
    base_mrp = random.choice([4999, 5999, 6999, 7999, 8999, 9999])
    products.append({
        "product_id": pid,
        "name": name,
        "category": random.choice(categories),
        "gender": random.choice(genders),
        "base_mrp": base_mrp,
        # image filename will be NIK001.jpg etc. in static/img
        "image_url": f"/static/img/{pid}.jpg",
        "color": random.choice(colors)
    })

df_products = pd.DataFrame(products)
df_products.to_csv(os.path.join(OUTPUT_DIR, "products.csv"), index=False)
print(f"Saved {len(df_products)} products to products.csv")

# -------------------------
# 3. Create global signals_daily.csv (social/news/stock)
# -------------------------

signals = []
for d in DATE_RANGE:
    day_idx = (d - START_DATE).days

    # A bit more complex patterns
    social = 0.4 * np.sin(day_idx / 20) + 0.2 * np.sin(day_idx / 60) + np.random.normal(0, 0.1)
    news = 0.3 * np.cos(day_idx / 25) + np.random.normal(0, 0.1)
    stock_trend = 0.02 * np.sin(day_idx / 90) + np.random.normal(0, 0.01)

    social = float(np.clip(social, -1, 1))
    news = float(np.clip(news, -1, 1))

    signals.append({
        "date": d.date(),
        "social_sentiment": social,
        "news_sentiment": news,
        "stock_trend": stock_trend
    })

df_signals = pd.DataFrame(signals)
df_signals.to_csv(os.path.join(OUTPUT_DIR, "signals_daily.csv"), index=False)
print(f"Saved daily signals to signals_daily.csv")

# -------------------------
# 4. Create price_history.csv with diverse patterns
# -------------------------

scenarios = [
    "steady_discount",    # discount slowly increases over time
    "festival_spikes",    # mostly stable, big drops near festivals
    "clearance_late",     # big discount only in last 6 months
    "hype_then_drop",     # high price early, then discounts after mid-2025
]

price_rows = []

def scenario_discount_adjustment(scenario, d, base_mrp, base_discount, product_phase):
    """
    Returns extra discount based on scenario and date.
    product_phase is a normalized time [0,1] over the 2-year period.
    """
    extra = 0.0

    if scenario == "steady_discount":
        # Gradually increase discount over time
        extra += 0.15 * product_phase

    elif scenario == "festival_spikes":
        # Mostly small discount, big extra near festivals
        if close_to_festival(d):
            extra += 0.25
        else:
            extra += 0.02 * np.sin((d - START_DATE).days / 40)

    elif scenario == "clearance_late":
        # Normal discounts first year, big clearance in last 6 months
        if d >= datetime(2026, 7, 1):
            extra += 0.3
        elif d >= datetime(2026, 1, 1):
            extra += 0.1

    elif scenario == "hype_then_drop":
        # Less discount at the start, more discount after mid-2025
        if d >= datetime(2025, 9, 1):
            extra += 0.2

    return extra

def get_event_flag(date):
    if close_to_festival(date):
        return "festival_sale"
    # simple fake season-end: end of Mar, Jun, Sep, Dec
    if date.month in [3, 6, 9, 12] and date.day > 20:
        return "season_end"
    return "none"

for _, prod in df_products.iterrows():
    pid = prod["product_id"]
    base_mrp = prod["base_mrp"]

    # Each product gets a scenario
    scenario = random.choice(scenarios)

    # Each product gets a base rating
    base_rating_product = np.clip(np.random.normal(4.0, 0.3), 3.0, 5.0)

    for size in SIZES:
        # Size-specific popularity: more popular means higher demand, different discount behavior
        size_popularity = np.clip(np.random.normal(1.0, 0.2), 0.6, 1.4)

        # start inventory different per size
        inventory = random.randint(40, 120)

        # start reviews per size
        reviews_count = random.randint(0, 30)

        # base discount per size
        base_discount_mean = np.random.uniform(0.05, 0.25)

        for d in DATE_RANGE:
            weekday = d.weekday()  # 0=Mon
            event_flag = get_event_flag(d)

            day_idx = (d - START_DATE).days
            product_phase = day_idx / float(DAYS - 1)  # from 0.0 to 1.0

            sig = df_signals.loc[df_signals["date"] == d.date()].iloc[0]
            social = sig["social_sentiment"]
            news = sig["news_sentiment"]
            stock_trend = sig["stock_trend"]

            # Base discount + scenario + sentiment + size popularity
            discount = base_discount_mean

            # Weekend slight bump
            if weekday in [5, 6]:
                discount += 0.03

            # Scenario-based adjustment
            discount += scenario_discount_adjustment(scenario, d, base_mrp, base_discount_mean, product_phase)

            # Festival/season-end overlays
            if event_flag == "festival_sale":
                discount += 0.15
            elif event_flag == "season_end":
                discount += 0.08

            # Sentiment-based adjustment
            discount += 0.04 * max(social, 0)     # positive social sentiment -> more promo
            discount -= 0.03 * max(-news, 0)      # negative news -> maybe lower promos

            # Size popularity effect (popular sizes might have slightly less discount)
            discount -= 0.05 * (size_popularity - 1.0)

            # Add a bit of noise
            discount += np.random.normal(0, 0.01)

            discount = float(np.clip(discount, 0.0, 0.65))
            selling_price = base_mrp * (1 - discount)

            # Demand index based on discount, sentiment, and popularity
            demand_index = float(
                np.clip(
                    0.3 + 0.7 * discount + 0.2 * social + 0.1 * (size_popularity - 1.0) + np.random.normal(0, 0.05),
                    0,
                    1,
                )
            )

            # Inventory dynamics: sell some units depending on demand
            sold_today = int(max(0, np.random.poisson(2 * demand_index)))
            inventory = max(0, inventory - sold_today)

            # Occasionally restock (simulate new shipments)
            if inventory < 10 and random.random() < 0.05:
                restock = random.randint(20, 60)
                inventory += restock

            # Reviews accumulate over time, depending on demand
            new_reviews = max(0, np.random.poisson(1.5 * demand_index))
            reviews_count += new_reviews

            # rating shifts slightly with sentiment
            rating_today = base_rating_product + 0.2 * social + np.random.normal(0, 0.1)
            rating_today = float(np.clip(rating_today, 3.0, 5.0))

            # reason_label (rule-based for now)
            if event_flag == "festival_sale":
                reason_label = "festival_sale"
            elif event_flag == "season_end":
                reason_label = "season_end"
            elif inventory < 15 and discount > 0.35:
                reason_label = "clearance"
            else:
                reason_label = "normal"

            price_rows.append({
                "product_id": pid,
                "size": size,
                "date": d.date(),
                "mrp": base_mrp,
                "selling_price": round(selling_price, 2),
                "discount_pct": round(discount, 3),
                "inventory_level": inventory,
                "rating_avg": round(rating_today, 2),
                "num_reviews": reviews_count,
                "demand_index": round(demand_index, 3),
                "social_sentiment": round(social, 3),
                "news_sentiment": round(news, 3),
                "stock_trend": round(stock_trend, 4),
                "event_flag": event_flag,
                "reason_label": reason_label,
                "scenario": scenario,
            })

df_price = pd.DataFrame(price_rows)

# -------------------------
# 5. Add targets: is_good_buy, future_drop_pct
# -------------------------

df_price = df_price.sort_values(["product_id", "size", "date"])
df_price["is_good_buy"] = 0
df_price["future_drop_pct"] = 0.0

horizon_days = 60  # look ahead this many days for future min price

for (pid, size), grp in df_price.groupby(["product_id", "size"]):
    grp = grp.sort_values("date")
    prices = grp["selling_price"].values
    dates = grp["date"].values.astype("datetime64[D]")  # ensure numpy datetime

    drops = np.zeros(len(grp))

    # 1) Compute future_drop_pct for each day
    for i in range(len(grp)):
        current_price = prices[i]
        current_date = dates[i]

        future_mask = (dates > current_date) & (
            dates <= current_date + np.timedelta64(horizon_days, "D")
        )

        if not future_mask.any():
            drops[i] = 0.0
            continue

        future_prices = prices[future_mask]
        min_future = future_prices.min()
        max_drop_pct = max(0.0, (current_price - min_future) / current_price)
        drops[i] = max_drop_pct

    # 2) Dynamic threshold per series:
    #    Mark "good buy" points as those with relatively LOW future_drop_pct
    #    e.g., lowest 30% of drops for this (product, size)
    if (drops > 0).any():
        # consider only non-zero drops to compute a threshold
        non_zero_drops = drops[drops > 0]
        if len(non_zero_drops) > 0:
            dynamic_thresh = np.quantile(non_zero_drops, 0.30)  # 30% percentile
        else:
            dynamic_thresh = 0.0
    else:
        dynamic_thresh = 0.0

    labels = (drops <= dynamic_thresh).astype(int)

    # Ensure at least 1 "good buy" exists: pick the absolute cheapest day
    if labels.sum() == 0:
        idx_min_price = prices.argmin()
        labels[idx_min_price] = 1

    df_price.loc[grp.index, "future_drop_pct"] = np.round(drops, 3)
    df_price.loc[grp.index, "is_good_buy"] = labels

# --- NEW: post-process latest rows to guarantee a mix of good/bad buys ---

# Ensure some product-size combinations are labelled as good buys
df_price = df_price.sort_values(["product_id", "size", "date"])

# Get index of the latest row for each (product_id, size)
latest_idx = df_price.groupby(["product_id", "size"]).tail(1).index

for idx in latest_idx:
    row = df_price.loc[idx]
    disc = float(row["discount_pct"])          # current discount fraction (0–0.65)
    drop = float(row["future_drop_pct"])       # expected future drop fraction

    # Very high discount and very little expected future drop -> force GOOD BUY
    if disc > 0.35 and drop < 0.05:
        df_price.at[idx, "is_good_buy"] = 1

    # Very low discount and large expected future drop -> force NOT good buy
    elif disc < 0.15 and drop > 0.15:
        df_price.at[idx, "is_good_buy"] = 0

    else:
        # For "in-between" cases, randomly mark about 40% as good buys
        df_price.at[idx, "is_good_buy"] = int(np.random.rand() < 0.4)

# -------------------------
# 6. Create reviews.csv (product-level)
# -------------------------

review_texts = [
    "Great cushioning and comfort for daily runs.",
    "Very stylish, but the fit is a bit tight.",
    "Good value for money during sale.",
    "Sole feels durable, happy with the purchase.",
    "Color is slightly different from images.",
    "Perfect for long distance running.",
    "Lightweight and breathable, highly recommend.",
    "Grip could be better on wet surfaces.",
    "Excellent for everyday wear.",
    "Superb stability and support."
]

review_rows = []
review_id = 1
for _, prod in df_products.iterrows():
    pid = prod["product_id"]
    # Each product gets 30 random reviews across the 2 years
    for _ in range(30):
        d = random.choice(DATE_RANGE).date()
        rating = random.randint(3, 5)
        sentiment = float(np.clip(0.2 * (rating - 3) + np.random.normal(0, 0.3), -1, 1))
        text = random.choice(review_texts)
        review_rows.append({
            "review_id": f"R{review_id:05d}",
            "product_id": pid,
            "date": d,
            "rating": rating,
            "sentiment": round(sentiment, 3),
            "text": text
        })
        review_id += 1

df_reviews = pd.DataFrame(review_rows)
df_reviews.to_csv(os.path.join(OUTPUT_DIR, "reviews.csv"), index=False)
print(f"Saved {len(df_reviews)} reviews to reviews.csv")

print("✅ Synthetic Smart Sneaks dataset (richer, 2-year, scenario-based) generated.")