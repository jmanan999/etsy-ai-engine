import json
import os

import pandas as pd

_OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_data() -> tuple[list, pd.DataFrame]:
    with open(os.path.join(_OUTPUTS, "image_clusters.json")) as f:
        clusters = json.load(f)["clusters"]

    agg = pd.read_csv(os.path.join(_OUTPUTS, "aggregated_products.csv"))
    agg["_title_key"] = agg["title"].str.lower().str.strip()
    return clusters, agg


def _lookup(title: str, agg: pd.DataFrame) -> tuple[float, float] | None:
    """Return (total_quantity, avg_price) for a title, or None if not found."""
    key = title.lower().strip()
    row = agg[agg["_title_key"] == key]
    if not row.empty:
        return float(row.iloc[0]["total_quantity"]), float(row.iloc[0]["avg_price"])

    # Fallback: partial word overlap match
    title_words = set(key.split())
    best_score, best_row = 0.0, None
    for _, r in agg.iterrows():
        agg_words = set(r["_title_key"].split())
        overlap = len(title_words & agg_words) / len(title_words | agg_words) if title_words | agg_words else 0
        if overlap > best_score:
            best_score, best_row = overlap, r

    if best_score >= 0.5 and best_row is not None:
        return float(best_row["total_quantity"]), float(best_row["avg_price"])
    return None


def _generate_insights(items_with_data: list[dict]) -> list[str]:
    """Rule-based insight generation from cluster item data."""
    insights = []
    ranked = sorted(items_with_data, key=lambda x: x["quantity"], reverse=True)
    best = ranked[0]
    worst = ranked[-1]

    stores = [i["store"] for i in items_with_data]
    prices = [i["price"] for i in items_with_data]
    quantities = [i["quantity"] for i in items_with_data]

    price_spread = max(prices) - min(prices)
    qty_spread = max(quantities) - min(quantities) if len(quantities) > 1 else 0

    # Dominance
    if len(stores) > 1 and best["quantity"] >= 2 * (quantities[1] if len(quantities) > 1 else 1):
        insights.append(f"'{best['store']}' dominates this product with {best['quantity']:.0f} units sold.")

    # Pricing vs performance
    if len(items_with_data) >= 2 and price_spread > 2:
        if best["price"] < worst["price"]:
            insights.append("Lower-priced listing outperforms — price sensitivity is high for this product.")
        elif best["price"] > worst["price"]:
            insights.append("Higher-priced listing wins — premium positioning is working here.")

    # Pricing spread
    if price_spread <= 2:
        insights.append("All stores use similar pricing — little competitive differentiation on price.")
    elif price_spread >= 10:
        insights.append(f"Wide price spread (${min(prices):.0f}–${max(prices):.0f}) — pricing strategy varies significantly.")

    # Multi-store competition
    if len(stores) >= 4:
        insights.append(f"High competition — {len(stores)} stores selling this product.")
    elif len(stores) == 2:
        insights.append("Direct head-to-head competition between 2 stores.")

    # Volume signal
    total_qty = sum(quantities)
    if total_qty >= 500:
        insights.append(f"High demand product — {total_qty:.0f} combined units across stores.")

    return insights if insights else ["Insufficient data for actionable insight."]


def analyze_clusters(clusters: list, agg: pd.DataFrame) -> list[dict]:
    results = []

    for cluster in clusters:
        items = cluster["items"]
        stores = list({i["store"] for i in items})
        if len(stores) < 2:
            continue

        items_with_data = []
        for item in items:
            match = _lookup(item["title"], agg)
            if match:
                qty, price = match
                items_with_data.append({
                    "store": item["store"],
                    "title": item["title"],
                    "quantity": qty,
                    "price": price,
                    "image_url": item["image_url"],
                })

        if not items_with_data:
            continue

        ranked = sorted(items_with_data, key=lambda x: x["quantity"], reverse=True)
        prices = [i["price"] for i in items_with_data]
        total_qty = sum(i["quantity"] for i in items_with_data)

        results.append({
            "cluster_id": cluster["cluster_id"],
            "num_stores": len(stores),
            "stores": stores,
            "best_store": ranked[0]["store"],
            "best_title": ranked[0]["title"],
            "best_quantity": round(ranked[0]["quantity"], 1),
            "total_quantity": round(total_qty, 1),
            "price_range": [round(min(prices), 2), round(max(prices), 2)],
            "avg_price": round(sum(prices) / len(prices), 2),
            "price_spread": round(max(prices) - min(prices), 2),
            "items": items_with_data,
            "insights": _generate_insights(items_with_data),
        })

    return sorted(results, key=lambda x: x["total_quantity"], reverse=True)


def save_insights(results: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump({"clusters": results}, f, indent=2)


def run():
    print("Loading data...")
    clusters, agg = load_data()

    print("Analyzing clusters...")
    results = analyze_clusters(clusters, agg)

    output_path = os.path.abspath(os.path.join(_OUTPUTS, "cluster_insights.json"))
    save_insights(results, output_path)

    print(f"\nAnalyzed {len(results)} multi-store clusters.\n")
    print("=" * 65)
    print("TOP 5 CLUSTERS BY COMBINED DEMAND")
    print("=" * 65)

    for r in results[:5]:
        print(f"\nCluster {r['cluster_id']} | {r['num_stores']} stores | "
              f"${r['price_range'][0]}–${r['price_range'][1]} | "
              f"{r['total_quantity']:.0f} total units")
        print(f"  Best store:  {r['best_store']}  ({r['best_quantity']:.0f} units @ ${r['avg_price']})")
        print(f"  Best title:  {r['best_title'][:75]}")
        for insight in r["insights"][:2]:
            print(f"  → {insight}")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    run()
