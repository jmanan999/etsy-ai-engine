import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from loader import load_google_sheet
from cleaner import clean_data, aggregate_data
from insights import load_aggregated_data, extract_keywords, extract_bigrams, save_insights

SHEET_URL = "https://docs.google.com/spreadsheets/d/10G9EOz16NknPMEA0cpwT-xRvaDe63HVOyWxLjLy28rw/edit?usp=sharing"
OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    print("Loading sheet...")
    df = load_google_sheet(SHEET_URL)

    print("Cleaning data...")
    cleaned = clean_data(df)
    cleaned_path = os.path.abspath(os.path.join(OUTPUTS, "cleaned_data.csv"))
    cleaned.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned data to {cleaned_path}")

    print("\nAggregating data...")
    aggregated = aggregate_data(cleaned)
    agg_path = os.path.abspath(os.path.join(OUTPUTS, "aggregated_products.csv"))
    aggregated.to_csv(agg_path, index=False)
    print(f"Saved aggregated data to {agg_path}")

    print("\nExtracting insights...")
    insights_path = os.path.abspath(os.path.join(OUTPUTS, "insights.json"))
    keywords = extract_keywords(aggregated)
    bigrams = extract_bigrams(aggregated)
    save_insights(keywords, bigrams, insights_path)
    print(f"Saved insights to {insights_path}")

    print("\nTop 10 keywords:")
    for word, count in keywords[:10]:
        print(f"  {word}: {count}")
    print("\nTop 10 bigrams:")
    for bigram, count in bigrams[:10]:
        print(f"  {bigram}: {count}")

    print("\nSample cleaned output:")
    print(cleaned.head(3).to_string())
    print("\nTop 5 products by quantity:")
    print(aggregated.head(5).to_string())


if __name__ == "__main__":
    main()
