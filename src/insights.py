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


def _tokenize(title: str) -> list[str]:
    words = re.findall(r"[a-z]+", title.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def extract_keywords(df: pd.DataFrame, top_n: int = 50) -> list[tuple[str, int]]:
    top = df.sort_values("total_quantity", ascending=False).head(50)
    counts: Counter = Counter()
    for title in top["title"]:
        counts.update(_tokenize(str(title)))
    return counts.most_common(top_n)


def extract_bigrams(df: pd.DataFrame, top_n: int = 30) -> list[tuple[str, int]]:
    top = df.sort_values("total_quantity", ascending=False).head(50)
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
