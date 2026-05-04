import math
import re

import pandas as pd

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _sheet_to_period(sheet_name: str) -> int | None:
    """Convert 'Items - May 2026' to a numeric period like 202605."""
    m = re.search(r"(\w+)\s+(\d{4})$", sheet_name)
    if not m:
        return None
    month = _MONTHS.get(m.group(1).lower())
    year = int(m.group(2))
    return year * 100 + month if month else None


_DECAY_K = 0.02  # per-month decay constant (~2% drop per month)


def _period_to_months(period: int) -> int:
    """Convert YYYYMM to total months (e.g. 202605 → 24277)."""
    return (period // 100) * 12 + (period % 100)


def _build_recency_weights(df: pd.DataFrame) -> dict[str, float]:
    """Map each sheet name to an exponential decay weight; latest month = 1.0."""
    sheets = df["_sheet"].dropna().unique()
    periods = {s: _sheet_to_period(s) for s in sheets}
    valid = {s: p for s, p in periods.items() if p}
    if not valid:
        return {s: 1.0 for s in sheets}

    latest_sheet = max(valid, key=valid.get)
    latest_months = _period_to_months(valid[latest_sheet])
    print(f"Latest sheet detected: '{latest_sheet}'")

    weights = {}
    for sheet, period in valid.items():
        diff = latest_months - _period_to_months(period)  # months behind latest
        weights[sheet] = round(math.exp(-_DECAY_K * diff), 4)

    # Debug: show weights for latest, ~12 months ago, ~36 months ago
    by_age = sorted(valid.items(), key=lambda x: x[1], reverse=True)
    targets = {"latest": 0, "1 year ago": 12, "3 years ago": 36}
    print("Sample weights:")
    for label, target_diff in targets.items():
        closest = min(by_age, key=lambda x: abs(latest_months - _period_to_months(x[1]) - target_diff))
        print(f"  {label} ({closest[0]}): {weights[closest[0]]}")

    return weights


_SKIP_PHRASES = ["processing time", "shipping upgrade", "rush order", "gift wrap"]

_RENAME = {
    "Item Title": "title",
    "Item Price": "price",
    "Quantity": "quantity",
    "Item Specs": "specs",
    "Order ID": "order_id",
    "Transaction ID": "transaction_id",
}


def _clean_title(title: str) -> str:
    title = title.lower().strip()
    # Remove leading digits glued to a word, e.g. "25ring" → "ring"
    title = re.sub(r"\b\d+([a-z])", r"\1", title)
    # Remove standalone numbers
    title = re.sub(r"\b\d+\b", "", title)
    # Normalize commas: ensure single space after each comma, no space before
    title = re.sub(r"\s*,\s*", ", ", title)
    # Collapse extra whitespace
    title = re.sub(r"\s+", " ", title).strip()
    # Remove duplicate comma-separated words (case-insensitive, preserve order)
    parts = [p.strip() for p in title.split(",")]
    seen = set()
    deduped = []
    for part in parts:
        key = part.lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(part)
    return ", ".join(deduped)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.rename(columns=_RENAME)

    df = df.dropna(subset=["title"])
    df = df[df["title"].astype(str).str.strip() != ""]

    # Remove non-product rows
    mask = df["title"].astype(str).str.lower().apply(
        lambda t: not any(phrase in t for phrase in _SKIP_PHRASES)
    )
    df = df[mask]

    df["title"] = df["title"].astype(str).apply(_clean_title)

    if "price" in df.columns:
        df["price"] = df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    df = df.drop(columns=["Thumbnail"], errors="ignore")
    df = df.reset_index(drop=True)
    print(f"Cleaned data: {len(df)} rows remaining.")
    return df


def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Group by title; use recency-weighted quantity, average price."""
    df = df.copy()

    if "_sheet" in df.columns:
        weights = _build_recency_weights(df)
        df["recency_weight"] = df["_sheet"].map(weights).fillna(1.0)
    else:
        df["recency_weight"] = 1.0

    df["weighted_quantity"] = df["quantity"] * df["recency_weight"]

    agg = (
        df.groupby("title", sort=False)
        .agg(
            total_quantity=("weighted_quantity", "sum"),
            avg_price=("price", "mean"),
        )
        .reset_index()
    )
    agg["total_quantity"] = agg["total_quantity"].round(2)
    agg["avg_price"] = agg["avg_price"].round(2)
    agg = agg.sort_values("total_quantity", ascending=False).reset_index(drop=True)
    print(f"Aggregated data: {len(agg)} unique products.")
    return agg
