import json
import os

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

_OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "outputs")
_SIMILARITY_THRESHOLD = 0.85
_TOP_N = 2000

_CATEGORY_KEYWORDS = {
    "ring":      ["ring", "band", "signet", "dome", "knuckle", "thumb"],
    "earring":   ["earring", "earrings", "hoop", "stud", "dangle", "drop", "huggie"],
    "bracelet":  ["bracelet", "bangle", "anklet", "ankle", "kada", "wristlet"],
    "necklace":  ["necklace", "pendant", "chain", "choker", "locket", "charm"],
    "cuff":      ["cuff", "armlet", "arm band", "armband", "upper arm"],
    "other":     [],
}


def get_category(title: str) -> str:
    t = title.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if category == "other":
            continue
        if any(kw in t for kw in keywords):
            return category
    return "other"


def get_price_bucket(price: float, low: float, high: float) -> str:
    if price <= low:
        return "low"
    if price <= high:
        return "medium"
    return "high"


def apply_blocking(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["title"].apply(get_category)

    low = df["price"].quantile(0.33)
    high = df["price"].quantile(0.66)
    df["price_bucket"] = df["price"].apply(lambda p: get_price_bucket(p, low, high))

    print("\n--- Blocking Summary ---")
    for cat, count in df["category"].value_counts().items():
        print(f"  Category: {cat:10s} → {count} items")
    print()
    for bucket, count in df["price_bucket"].value_counts().items():
        print(f"  Price bucket: {bucket:6s} → {count} items  (low≤${low:.0f}, med≤${high:.0f}, high>${high:.0f})")

    naive_comparisons = len(df) * (len(df) - 1) // 2
    blocked_comparisons = sum(
        len(g) * (len(g) - 1) // 2
        for _, g in df.groupby(["category", "price_bucket"])
    )
    print(f"\n  Naive comparisons:   {naive_comparisons:,}")
    print(f"  Blocked comparisons: {blocked_comparisons:,}")
    print(f"  Comparisons avoided: {naive_comparisons - blocked_comparisons:,}  "
          f"({100*(naive_comparisons-blocked_comparisons)/naive_comparisons:.1f}% reduction)")
    print("------------------------\n")

    return df


def load_data() -> pd.DataFrame:
    cleaned = pd.read_csv(os.path.join(_OUTPUTS, "cleaned_data.csv"),
                          dtype={"transaction_id": str})
    images = pd.read_csv(os.path.join(_OUTPUTS, "image_urls.csv"),
                         dtype={"Transaction ID": str})
    images = images.rename(columns={"Transaction ID": "transaction_id"})

    df = cleaned.merge(images, on="transaction_id", how="inner")
    df = df.dropna(subset=["image_url", "title", "store_name"])

    df = df.reset_index(drop=True)
    print(f"Products with images: {len(df)}")
    return df


def _load_image(url: str):
    try:
        url = url.replace("il_75x75", "il_794x794")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        from io import BytesIO
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def generate_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> tuple[list, list[int]]:
    """Fetch images and embed them; return (embeddings, valid_indices)."""
    embeddings, valid_idx = [], []
    for i, row in df.iterrows():
        img = _load_image(row["image_url"])
        if img is None:
            continue
        emb = model.encode(img, convert_to_numpy=True)
        embeddings.append(emb)
        valid_idx.append(i)
        if (len(valid_idx)) % 50 == 0:
            print(f"  Embedded {len(valid_idx)} images...")
    return embeddings, valid_idx


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compute_title_similarity(title1: str, title2: str) -> float:
    words1 = set(title1.lower().split())
    words2 = set(title2.lower().split())
    common = words1 & words2
    total = words1 | words2
    return len(common) / len(total) if total else 0.0


def _matches(emb_i: np.ndarray, title_i: str, emb_j: np.ndarray, title_j: str) -> tuple[bool, float, float, str]:
    """Return (is_match, img_sim, title_sim, condition_label)."""
    img_sim = _cosine_similarity(emb_i, emb_j)
    title_sim = compute_title_similarity(title_i, title_j)
    if img_sim >= 0.85:
        return True, img_sim, title_sim, "STRONG_IMAGE"
    if img_sim >= 0.75 and title_sim >= 0.4:
        return True, img_sim, title_sim, "HYBRID_MATCH"
    return False, img_sim, title_sim, ""


def cluster_by_similarity(embeddings: list, valid_idx: list, df: pd.DataFrame) -> list[dict]:
    """Greedy clustering with cluster-wide comparison (chain similarity)."""
    clusters = []        # list of list-of-items
    cluster_embs = []    # list of list-of-embeddings (parallel to clusters)
    assigned = set()

    for i, (emb_i, idx_i) in enumerate(zip(embeddings, valid_idx)):
        if i in assigned:
            continue
        title_i = df.at[idx_i, "title"]
        new_cluster = [{
            "title": title_i,
            "store": df.at[idx_i, "store_name"],
            "image_url": df.at[idx_i, "image_url"],
        }]
        new_embs = [emb_i]
        assigned.add(i)

        for j, (emb_j, idx_j) in enumerate(zip(embeddings, valid_idx)):
            if j in assigned:
                continue
            title_j = df.at[idx_j, "title"]
            # Compare j against every item already in the cluster
            for emb_k, item_k in zip(new_embs, new_cluster):
                matched, img_sim, title_sim, condition = _matches(
                    emb_k, item_k["title"], emb_j, title_j
                )
                if matched:
                    new_cluster.append({
                        "title": title_j,
                        "store": df.at[idx_j, "store_name"],
                        "image_url": df.at[idx_j, "image_url"],
                        "img_sim": round(img_sim, 3),
                        "title_sim": round(title_sim, 3),
                        "condition": condition,
                    })
                    new_embs.append(emb_j)
                    assigned.add(j)
                    break

        clusters.append(new_cluster)
        cluster_embs.append(new_embs)

    return clusters


def save_clusters(clusters: list[list], output_path: str) -> None:
    data = {
        "clusters": [
            {"cluster_id": f"P{str(i+1).zfill(3)}", "items": items}
            for i, items in enumerate(clusters)
        ]
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def run():
    import time
    start = time.time()

    print("Loading data...")
    df = load_data()
    sample = df.head(_TOP_N).reset_index(drop=True)
    print(f"Using top {len(sample)} products for clustering.")

    sample = apply_blocking(sample)

    print("Loading CLIP model...")
    model = SentenceTransformer("clip-ViT-B-32")

    all_clusters = []
    groups = list(sample.groupby(["category", "price_bucket"]))
    for (cat, bucket), block in groups:
        block = block.reset_index(drop=True)
        print(f"Block [{cat} / {bucket}]: {len(block)} items — embedding...")
        embeddings, valid_idx = generate_embeddings(block, model)
        if not embeddings:
            continue
        clusters = cluster_by_similarity(embeddings, valid_idx, block)
        all_clusters.extend(clusters)
        print(f"  → {len(clusters)} clusters formed.")

    output_path = os.path.abspath(os.path.join(_OUTPUTS, "image_clusters.json"))
    save_clusters(all_clusters, output_path)

    multi_store = [c for c in all_clusters if len(set(i["store"] for i in c)) > 1]
    elapsed = round(time.time() - start, 1)

    print(f"\nTotal clusters:            {len(all_clusters)}")
    print(f"Multi-store clusters:      {len(multi_store)}")
    print(f"Runtime:                   {elapsed}s")

    if multi_store:
        print("\nSample multi-store cluster:")
        for item in multi_store[0][:5]:
            print(f"  Store:     {item['store']}")
            print(f"  Title:     {item['title'][:90]}")
            print(f"  Image:     {item['image_url']}")
            if "condition" in item:
                print(f"  img_sim={item['img_sim']}  title_sim={item['title_sim']}  [{item['condition']}]")
            print()

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    run()
