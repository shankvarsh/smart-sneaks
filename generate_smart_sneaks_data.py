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
DAYS = 180

START_DATE = datetime.today() - timedelta(days=DAYS)
DATE_RANGE = [START_DATE + timedelta(days=i) for i in range(DAYS)]

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Some fake sale/festival dates (relative to START_DATE)
festival_offsets = [30, 90, 150]  # 3 big sales in 180 days
festival_dates = [START_DATE + timedelta(days=o) for o in festival_offsets]

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
        "image_url": f"https://via.placeholder.com/300x300.png?text={name.replace(' ', '+')}",
        "color": random.choice(colors)
    })

df_products = pd.DataFrame(products)
df_products.to_csv(os.path.join(OUTPUT_DIR, "products.csv"), index=False)
print(f"Saved {len(df_products)} products to products.csv")

# -------------------------
# 3. Create signals_daily.csv
# -------------------------

signals = []
for d in DATE_RANGE:
    # Simulate some smooth trends using sin waves + noise
    day_idx = (d - START_DATE).days
    social = 0.3 * np.sin(day_idx / 15) + np.random.normal(0, 0.1)
    news = 0.2 * np.cos(day_idx / 20) + np.random.normal(0, 0.1)
    stock_trend = 0.01 * np.sin(day_idx / 30) + np.random.normal(0, 0.005)

    # Clip to reasonable ranges
    social = float(np.clip(social, -1, 1))
    news = float(np.clip(news, -1, 1))
    stock_trend = float(stock_trend)
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
# 4. Create price_history.csv
# -------------------------

price_rows = []

def get_event_flag(date):
    for fd in festival_dates:
        if abs((date - fd).days) <= 5:   # ±5 days around festival → sale
            return "festival_sale"
    # simple season-end: every ~60 days window end
    if date.day in [10, 20]:  # purely synthetic
        return "season_end"
    return "none"

for _, prod in df_products.iterrows():
    pid = prod["product_id"]
    base_mrp = prod["base_mrp"]

    for size in SIZES:
        # size-specific bias
        size_popularity = 1.0 + np.random.normal(0, 0.1)  # demand factor

        # start inventory high, decay over time with randomness
        inventory = 100 + random.randint(-10, 10)

        # random base discount range for this product-size
        base_discount_mean = np.random.uniform(0.05, 0.20)

        for d in DATE_RANGE:
            weekday = d.weekday()  # 0=Mon
            event_flag = get_event_flag(d)

            # Get signals
            sig = df_signals.loc[df_signals["date"] == d.date()].iloc[0]
            social = sig["social_sentiment"]
            news = sig["news_sentiment"]
            stock_trend = sig["stock_trend"]

            # Compute discount based on:
            # - base discount
            # - festival / sale multiplier
            # - sentiment / news
            discount = base_discount_mean

            # Weekend a bit higher discount
            if weekday in [5, 6]:  # Sat, Sun
                discount += 0.03

            # Festival sales get bigger discounts
            if event_flag == "festival_sale":
                discount += 0.20
            elif event_flag == "season_end":
                discount += 0.10

            # sentiment-driven adjustment
            discount += 0.05 * max(social, 0)   # positive social sentiment → promo
            discount -= 0.03 * max(-news, 0)    # bad news may reduce promo

            # Bound discount
            discount = float(np.clip(discount, 0.0, 0.6))

            selling_price = base_mrp * (1 - discount)

            # Simulate inventory drop if demand high (proxied by discount + social)
            demand_index = float(np.clip(0.5 + discount + social + np.random.normal(0, 0.1), 0, 1))
            inventory_change = -int(1 + 5 * demand_index * random.random())
            inventory = max(0, inventory + inventory_change)

            # basic rating trend: more demand → better ratings over time
            rating_base = np.clip(3.5 + 0.5 * social + np.random.normal(0, 0.3), 1, 5)
            num_reviews = max(0, int((DAYS - (d - START_DATE).days) * demand_index * random.random()))

            # reason_label (for now rules, you will train classifier later)
            if event_flag == "festival_sale":
                reason_label = "festival_sale"
            elif event_flag == "season_end":
                reason_label = "season_end"
            elif inventory < 10 and discount > 0.3:
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
                "rating_avg": round(rating_base, 2),
                "num_reviews": num_reviews,
                "demand_index": round(demand_index, 3),
                "social_sentiment": round(social, 3),
                "news_sentiment": round(news, 3),
                "event_flag": event_flag,
                "reason_label": reason_label
            })

df_price = pd.DataFrame(price_rows)

# -------------------------
# 5. Add targets: is_good_buy, future_drop_pct
# -------------------------

df_price = df_price.sort_values(["product_id", "size", "date"])

df_price["is_good_buy"] = 0
df_price["future_drop_pct"] = 0.0

horizon = 30  # days

for (pid, size), grp in df_price.groupby(["product_id", "size"]):
    grp = grp.sort_values("date")
    prices = grp["selling_price"].values
    dates = grp["date"].values

    labels = np.zeros(len(grp), dtype=int)
    drops = np.zeros(len(grp))

    for i in range(len(grp)):
        current_price = prices[i]
        current_date = dates[i]
        # look ahead up to horizon days
        future_mask = (dates > current_date) & (dates <= current_date + np.timedelta64(horizon, 'D'))
        if not future_mask.any():
            labels[i] = 0
            drops[i] = 0.0
            continue
        future_prices = prices[future_mask]
        min_future = future_prices.min()
        max_drop_pct = max(0.0, (current_price - min_future) / current_price)

        # good buy if we are within 5% of min future price
        if current_price <= 1.05 * min_future:
            labels[i] = 1
        drops[i] = max_drop_pct

    df_price.loc[grp.index, "is_good_buy"] = labels
    df_price.loc[grp.index, "future_drop_pct"] = np.round(drops, 3)

df_price.to_csv(os.path.join(OUTPUT_DIR, "price_history.csv"), index=False)
print(f"Saved {len(df_price)} rows to price_history.csv")

# -------------------------
# 6. Create simple reviews.csv
# -------------------------

review_texts = [
    "Great cushioning and comfort for daily runs.",
    "Very stylish, but the fit is a bit tight.",
    "Good value for money during sale.",
    "Sole feels durable, happy with the purchase.",
    "Color is slightly different from images.",
    "Perfect for long distance running.",
    "Lightweight and breathable, highly recommend."
]

review_rows = []
review_id = 1
for _, prod in df_products.iterrows():
    pid = prod["product_id"]
    # Each product gets 20 random reviews across time
    for _ in range(20):
        d = random.choice(DATE_RANGE).date()
        rating = random.randint(3, 5)
        sentiment = float(np.clip(0.2 * (rating - 3) + np.random.normal(0, 0.3), -1, 1))
        text = review_texts[random.randint(0, len(review_texts)-1)]
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

print("✅ Synthetic Smart Sneaks dataset generated.")