import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from loader import load_google_sheet
from cleaner import clean_data

SHEET_URL = "https://docs.google.com/spreadsheets/d/10G9EOz16NknPMEA0cpwT-xRvaDe63HVOyWxLjLy28rw/edit?usp=sharing"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "cleaned_data.csv")


def main():
    print("Loading sheet...")
    df = load_google_sheet(SHEET_URL)

    print("Cleaning data...")
    cleaned = clean_data(df)

    out = os.path.abspath(OUTPUT_PATH)
    cleaned.to_csv(out, index=False)
    print(f"Saved to {out}")
    print("\nSample output:")
    print(cleaned.head(5).to_string())


if __name__ == "__main__":
    main()
