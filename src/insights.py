import json
import re
from collections import Counter

import pandas as pd

_STOPWORDS = {
    "and", "for", "with", "the", "a", "an", "of", "in", "to", "on",
    "at", "by", "or", "is", "it", "as", "be", "my", "your", "our",
    "from", "this", "that", "are", "was", "but", "not", "so", "do",
    "i", "you", "we", "he", "she", "they", "its", "also",
}


def load_aggregated_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


_TOP_N_PRODUCTS = 300


def _tokenize(title: str) -> list[str]:
    # Only alphabetic words, length >= 3, not stopwords
    words = re.findall(r"[a-z]{3,}", title.lower())
    return [w for w in words if w not in _STOPWORDS]


def extract_keywords(df: pd.DataFrame, top_n: int = 50) -> list[tuple[str, int]]:
    print(f"Total products in dataset: {len(df)}")
    top = df.sort_values("total_quantity", ascending=False).head(_TOP_N_PRODUCTS)
    print(f"Products used for analysis: {len(top)}")
    counts: Counter = Counter()
    for title in top["title"]:
        counts.update(_tokenize(str(title)))
    return counts.most_common(top_n)


def extract_bigrams(df: pd.DataFrame, top_n: int = 30) -> list[tuple[str, int]]:
    top = df.sort_values("total_quantity", ascending=False).head(_TOP_N_PRODUCTS)
    counts: Counter = Counter()
    for title in top["title"]:
        words = _tokenize(str(title))
        counts.update(f"{words[i]} {words[i+1]}" for i in range(len(words) - 1))
    return counts.most_common(top_n)


def save_insights(keywords: list, bigrams: list, output_path: str) -> None:
    data = {
        "top_keywords": [{"word": w, "count": c} for w, c in keywords],
        "top_bigrams": [{"bigram": b, "count": c} for b, c in bigrams],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


_CATEGORIES = {
    "MATERIAL": {"gold", "silver", "brass", "gemstone"},
    "PRODUCT":  {"ring", "earrings", "necklace", "bracelet", "pendant", "cuff"},
    "STYLE":    {"boho", "statement", "minimalist", "vintage", "gothic", "stacking"},
    "OCCASION": {"gift", "wedding", "birthday", "anniversary"},
}

# Reverse lookup: word -> category
_WORD_TO_CATEGORY = {
    word: cat
    for cat, words in _CATEGORIES.items()
    for word in words
}


def _title_pattern(title: str) -> str | None:
    """Return the ordered set of categories present in a title, e.g. 'MATERIAL + PRODUCT'."""
    seen = []
    for word in _tokenize(title):
        cat = _WORD_TO_CATEGORY.get(word)
        if cat and cat not in seen:
            seen.append(cat)
    return " + ".join(seen) if seen else None


def extract_patterns(df: pd.DataFrame, top_n: int = 20) -> list[tuple[str, int]]:
    top = df.sort_values("total_quantity", ascending=False).head(_TOP_N_PRODUCTS)
    counts: Counter = Counter()
    for title in top["title"]:
        pattern = _title_pattern(str(title))
        if pattern:
            counts[pattern] += 1
    return counts.most_common(top_n)


def save_patterns(patterns: list, output_path: str) -> None:
    data = {"top_patterns": [{"pattern": p, "count": c} for p, c in patterns]}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
