import pandas as pd
import re


def load_google_sheet(url: str) -> pd.DataFrame:
    """Load a public Google Sheet as a DataFrame."""
    csv_url = _to_csv_url(url)
    try:
        df = pd.read_csv(csv_url)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load sheet: {e}")


def _to_csv_url(url: str) -> str:
    """Convert a Google Sheets share URL to a CSV export URL."""
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("Invalid Google Sheets URL.")
    sheet_id = match.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
