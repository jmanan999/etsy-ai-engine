import pandas as pd


# Rows whose title contains these strings are not real products
_SKIP_PHRASES = ["processing time", "shipping upgrade", "rush order", "gift wrap"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows with missing title
    df = df.dropna(subset=["title"])
    df = df[df["title"].astype(str).str.strip() != ""]

    # Remove non-product rows
    mask = df["title"].astype(str).str.lower().apply(
        lambda t: not any(phrase in t for phrase in _SKIP_PHRASES)
    )
    df = df[mask]

    # Normalize title
    df["title"] = df["title"].astype(str).str.lower().str.strip()
    df["title"] = df["title"].str.replace(r"\s+", " ", regex=True)

    # Convert price to numeric
    if "price" in df.columns:
        df["price"] = (
            df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Convert orders / quantity to numeric
    for col in ["orders", "quantity", "sales"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)
    print(f"Cleaned data: {len(df)} rows remaining.")
    return df
