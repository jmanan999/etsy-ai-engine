import re
from datetime import datetime

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


def _build_recency_weights(df: pd.DataFrame) -> dict[str, float]:
    """Map each sheet name to a linear recency weight in (0, 1]."""
    sheets = df["_sheet"].dropna().unique()
    periods = {s: _sheet_to_period(s) for s in sheets}
    valid = {s: p for s, p in periods.items() if p}
    if not valid:
        return {s: 1.0 for s in sheets}

    min_p, max_p = min(valid.values()), max(valid.values())
    latest = max(valid, key=valid.get)
    print(f"Latest sheet detected: '{latest}' (period {max_p})")

    weights = {}
    for sheet, period in valid.items():
        if max_p == min_p:
            weights[sheet] = 1.0
        else:
            weights[sheet] = round(0.1 + 0.9 * (period - min_p) / (max_p - min_p), 4)

    # Print a few sample weights
    sample = sorted(weights.items(), key=lambda x: x[1])
    print("Sample weights (oldest → newest):")
    for s, w in sample[:3] + sample[-3:]:
        print(f"  {s}: {w}")

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
