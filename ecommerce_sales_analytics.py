"""
E-Commerce Sales Performance Analytics
Author: Student Project
Purpose: Demonstrate an end-to-end Data Analytics workflow:
data generation/loading -> cleaning -> KPI analysis -> visualization -> insights.

The project uses a reproducible synthetic e-commerce dataset so it can run
without proprietary data or external files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

def create_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    categories = ["Electronics", "Home & Kitchen", "Fashion", "Beauty", "Sports"]
    regions = ["North", "South", "East", "West"]
    channels = ["Website", "Mobile App", "Marketplace"]
    products = {
        "Electronics": ["Wireless Headphones", "Smart Watch", "Bluetooth Speaker", "Power Bank"],
        "Home & Kitchen": ["Mixer Grinder", "Air Fryer", "Cookware Set", "Vacuum Cleaner"],
        "Fashion": ["Running Shoes", "Casual Shirt", "Backpack", "Jeans"],
        "Beauty": ["Face Serum", "Hair Dryer", "Skincare Kit", "Perfume"],
        "Sports": ["Yoga Mat", "Dumbbells", "Cricket Kit", "Fitness Band"],
    }
    base_price = {
        "Electronics": (1200, 9000), "Home & Kitchen": (800, 7000),
        "Fashion": (700, 4500), "Beauty": (500, 3500), "Sports": (900, 5000)
    }
    rows = []
    for i in range(n):
        cat = rng.choice(categories, p=[.28,.23,.22,.15,.12])
        region = rng.choice(regions, p=[.28,.25,.18,.29])
        channel = rng.choice(channels, p=[.42,.33,.25])
        product = rng.choice(products[cat])
        date = rng.choice(dates)
        low, high = base_price[cat]
        price = rng.uniform(low, high)
        qty = int(rng.choice([1,2,3,4,5], p=[.50,.27,.13,.07,.03]))
        discount = float(rng.choice([0,.05,.10,.15,.20], p=[.32,.24,.22,.16,.06]))
        revenue = price * qty * (1-discount)
        cost = revenue * rng.uniform(.62,.82)
        profit = revenue - cost
        rows.append([f"ORD{i+1:04d}", date, cat, product, region, channel,
                     qty, round(price,2), discount, round(revenue,2), round(profit,2)])
    return pd.DataFrame(rows, columns=[
        "Order_ID","Order_Date","Category","Product","Region","Channel",
        "Quantity","Unit_Price","Discount","Revenue","Profit"
    ])

def main():
    df = create_dataset()
    # Data quality checks
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df = df.drop_duplicates().dropna()
    df["Month"] = df["Order_Date"].dt.strftime("%b")
    df["Month_Num"] = df["Order_Date"].dt.month

    # KPI calculations
    revenue = df["Revenue"].sum()
    profit = df["Profit"].sum()
    orders = df["Order_ID"].nunique()
    aov = revenue / orders
    margin = profit / revenue * 100

    print("=== E-COMMERCE SALES ANALYTICS ===")
    print(f"Total Revenue : ₹{revenue:,.2f}")
    print(f"Total Profit  : ₹{profit:,.2f}")
    print(f"Orders        : {orders:,}")
    print(f"Average Order Value: ₹{aov:,.2f}")
    print(f"Profit Margin : {margin:.2f}%")

    monthly = df.groupby(["Month_Num","Month"], as_index=False).agg(
        Revenue=("Revenue","sum"), Profit=("Profit","sum")
    ).sort_values("Month_Num")
    category = df.groupby("Category", as_index=False).agg(
        Revenue=("Revenue","sum"), Profit=("Profit","sum")
    ).sort_values("Revenue", ascending=False)
    region = df.groupby("Region", as_index=False).agg(
        Revenue=("Revenue","sum"), Profit=("Profit","sum")
    ).sort_values("Revenue", ascending=False)
    channel = df.groupby("Channel", as_index=False).agg(
        Revenue=("Revenue","sum"), Orders=("Order_ID","nunique")
    ).sort_values("Revenue", ascending=False)

    # Export analysis tables
    df.to_csv(OUTPUT/"ecommerce_sales_data.csv", index=False)
    monthly.to_csv(OUTPUT/"monthly_summary.csv", index=False)
    category.to_csv(OUTPUT/"category_summary.csv", index=False)
    region.to_csv(OUTPUT/"region_summary.csv", index=False)
    channel.to_csv(OUTPUT/"channel_summary.csv", index=False)

    # Visualizations
    plt.figure(figsize=(9,5))
    plt.plot(monthly["Month"], monthly["Revenue"], marker="o")
    plt.title("Monthly Revenue Trend"); plt.xlabel("Month"); plt.ylabel("Revenue (₹)")
    plt.tight_layout(); plt.savefig(OUTPUT/"monthly_revenue.png", dpi=160); plt.show()

    plt.figure(figsize=(9,5))
    plt.bar(category["Category"], category["Revenue"])
    plt.title("Revenue by Category"); plt.xlabel("Category"); plt.ylabel("Revenue (₹)")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(OUTPUT/"category_revenue.png", dpi=160); plt.show()

    plt.figure(figsize=(8,5))
    plt.bar(region["Region"], region["Profit"])
    plt.title("Profit by Region"); plt.xlabel("Region"); plt.ylabel("Profit (₹)")
    plt.tight_layout(); plt.savefig(OUTPUT/"region_profit.png", dpi=160); plt.show()

    plt.figure(figsize=(8,5))
    plt.bar(channel["Channel"], channel["Revenue"])
    plt.title("Revenue by Sales Channel"); plt.xlabel("Channel"); plt.ylabel("Revenue (₹)")
    plt.tight_layout(); plt.savefig(OUTPUT/"channel_revenue.png", dpi=160); plt.show()

if __name__ == "__main__":
    main()
