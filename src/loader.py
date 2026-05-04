import re
import urllib.request
from urllib.parse import quote

import pandas as pd


def load_google_sheet(url: str) -> pd.DataFrame:
    sheet_id = _extract_sheet_id(url)
    sheet_names = _get_items_sheet_names(sheet_id)
    print(f"Found {len(sheet_names)} 'Items' sheets to load.")

    _EXPECTED_COLS = ["Order ID", "Transaction ID", "Item Title", "Item Specs", "Quantity", "Item Price", "Thumbnail"]

    frames = []
    for name in sheet_names:
        csv_url = (
            f"https://docs.google.com/spreadsheets/d/{sheet_id}"
            f"/gviz/tq?tqx=out:csv&sheet={quote(name)}"
        )
        try:
            df = pd.read_csv(csv_url)
            # Some sheets lack a header row — detect and fix
            if not set(_EXPECTED_COLS).issubset(set(df.columns)):
                df = pd.read_csv(csv_url, header=None, names=_EXPECTED_COLS)
            df["_sheet"] = name
            frames.append(df)
        except Exception as e:
            print(f"  Skipped '{name}': {e}")

    if not frames:
        raise RuntimeError("No sheets loaded.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined)} rows total across {len(frames)} sheets.")
    return combined


def _extract_sheet_id(url: str) -> str:
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("Invalid Google Sheets URL.")
    return match.group(1)


def _get_items_sheet_names(sheet_id: str) -> list[str]:
    """Scrape the spreadsheet HTML to find all 'Items - *' tab names."""
    page_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
    req = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        html = r.read().decode("utf-8")

    all_names = re.findall(r'docs-sheet-tab-caption">([^<]+)<', html)
    return [n for n in all_names if n.startswith("Items -")]
