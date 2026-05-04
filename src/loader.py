import re
import urllib.request
from urllib.parse import quote

import pandas as pd

_ITEMS_COLS = ["Order ID", "Transaction ID", "Item Title", "Item Specs", "Quantity", "Item Price", "Thumbnail"]


def load_google_sheet(url: str) -> pd.DataFrame:
    sheet_id = _extract_sheet_id(url)
    html = _fetch_html(sheet_id)
    all_names = re.findall(r'docs-sheet-tab-caption">([^<]+)<', html)

    items_names = [n for n in all_names if n.startswith("Items -")]
    orders_names = [n for n in all_names if n.startswith("Orders -")]
    print(f"Found {len(items_names)} 'Items' sheets and {len(orders_names)} 'Orders' sheets.")

    items_df = _load_items(sheet_id, items_names)
    orders_df = _load_orders(sheet_id, orders_names)

    combined = items_df.merge(orders_df, on="Order ID", how="left")
    matched = combined["store_name"].notna().sum()
    print(f"Join complete: {matched}/{len(combined)} rows matched to a store.")

    print("\nSample rows with store_name:")
    print(combined[["Order ID", "Item Title", "Quantity", "Item Price", "store_name"]].dropna(subset=["store_name"]).head(5).to_string())

    return combined


def _load_items(sheet_id: str, names: list[str]) -> pd.DataFrame:
    frames = []
    for name in names:
        url = _csv_url(sheet_id, name)
        try:
            df = pd.read_csv(url)
            if not set(_ITEMS_COLS).issubset(set(df.columns)):
                df = pd.read_csv(url, header=None, names=_ITEMS_COLS)
            df["_sheet"] = name
            frames.append(df)
        except Exception as e:
            print(f"  Skipped '{name}': {e}")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined)} item rows across {len(frames)} sheets.")
    return combined


def _load_orders(sheet_id: str, names: list[str]) -> pd.DataFrame:
    frames = []
    for name in names:
        url = _csv_url(sheet_id, name)
        try:
            df = pd.read_csv(url, usecols=["Order ID", "Store Name"])
            df = df.rename(columns={"Store Name": "store_name"})
            frames.append(df)
        except Exception as e:
            print(f"  Skipped '{name}': {e}")
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Order ID"])
    print(f"Loaded {len(combined)} unique orders across {len(frames)} sheets.")
    return combined


def _csv_url(sheet_id: str, sheet_name: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet_name)}"


def _fetch_html(sheet_id: str) -> str:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8")


def _extract_sheet_id(url: str) -> str:
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("Invalid Google Sheets URL.")
    return match.group(1)
