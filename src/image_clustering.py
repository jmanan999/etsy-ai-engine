import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

_embedding_cache: dict[str, np.ndarray] = {}

_OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "outputs")
_CLUSTERS_DIR = os.path.join(_OUTPUTS, "clusters")
_CACHE_PATH = os.path.abspath(os.path.join(_OUTPUTS, "embedding_cache.pkl"))
_SIMILARITY_THRESHOLD = 0.85

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


def load_embedding_cache() -> None:
    global _embedding_cache
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "rb") as f:
            _embedding_cache = pickle.load(f)
        print(f"Loaded embedding cache: {len(_embedding_cache)} entries from {_CACHE_PATH}")
    else:
        print("No embedding cache found — starting fresh.")


def save_embedding_cache() -> None:
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(_embedding_cache, f)
    print(f"Embedding cache saved: {len(_embedding_cache)} entries → {_CACHE_PATH}")


def generate_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> tuple[list, list[int]]:
    """Parallel image fetch + embed with per-URL caching."""
    cache_hits = 0

    def load_and_embed(args):
        nonlocal cache_hits
        idx, row = args
        url = row["image_url"]
        if url in _embedding_cache:
            cache_hits += 1
            return idx, _embedding_cache[url]
        img = _load_image(url)
        if img is None:
            return idx, None
        emb = model.encode(img, convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        _embedding_cache[url] = emb
        return idx, emb

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(load_and_embed, df.iterrows()))

    embeddings, valid_idx = [], []
    for idx, emb in results:
        if emb is not None:
            embeddings.append(emb)
            valid_idx.append(idx)

    elapsed = round(time.time() - t0, 1)
    print(f"  Embedded {len(embeddings)} images in {elapsed}s "
          f"(cache hits: {cache_hits}, new: {len(embeddings) - cache_hits})")
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


def _anchor_match(anchor_emb: np.ndarray, anchor_title: str,
                   emb_j: np.ndarray, title_j: str,
                   cluster_embs: list[np.ndarray]) -> tuple[bool, float, float, str]:
    """
    Accept j into cluster only if it matches the anchor directly.
    Primary:  anchor img_sim >= 0.85  (STRONG_IMAGE)
    Fallback: anchor img_sim >= 0.80 AND avg img_sim across cluster >= 0.75  (HYBRID_MATCH)
    """
    anchor_img_sim = _cosine_similarity(anchor_emb, emb_j)
    anchor_title_sim = compute_title_similarity(anchor_title, title_j)

    if anchor_img_sim >= 0.85:
        return True, anchor_img_sim, anchor_title_sim, "STRONG_IMAGE"

    if anchor_img_sim >= 0.80:
        avg_cluster_sim = sum(_cosine_similarity(e, emb_j) for e in cluster_embs) / len(cluster_embs)
        if avg_cluster_sim >= 0.75:
            return True, anchor_img_sim, anchor_title_sim, "HYBRID_MATCH"

    return False, anchor_img_sim, anchor_title_sim, ""


def cluster_by_similarity(embeddings: list, valid_idx: list, df: pd.DataFrame) -> list[dict]:
    """Greedy clustering with anchor-based validation to prevent chain drift."""
    clusters = []
    cluster_embs_list = []  # parallel to clusters: list of list-of-embeddings
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
        anchor_emb = emb_i
        anchor_title = title_i
        assigned.add(i)

        for j, (emb_j, idx_j) in enumerate(zip(embeddings, valid_idx)):
            if j in assigned:
                continue
            title_j = df.at[idx_j, "title"]
            matched, img_sim, title_sim, condition = _anchor_match(
                anchor_emb, anchor_title, emb_j, title_j, new_embs
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

        if len(new_cluster) > 200:
            print(f"  ⚠️  Large cluster warning: {len(new_cluster)} items (anchor: '{title_i[:60]}')")

        clusters.append(new_cluster)
        cluster_embs_list.append(new_embs)

    return clusters


def _block_output_path(cat: str, bucket: str) -> str:
    os.makedirs(_CLUSTERS_DIR, exist_ok=True)
    return os.path.join(_CLUSTERS_DIR, f"clusters_{cat}_{bucket}.json")


def _save_block(clusters: list[list], cat: str, bucket: str, id_offset: int) -> str:
    path = _block_output_path(cat, bucket)
    data = {
        "clusters": [
            {"cluster_id": f"P{str(id_offset + i + 1).zfill(4)}", "items": items}
            for i, items in enumerate(clusters)
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


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
    start = time.time()

    load_embedding_cache()

    print("Loading data...")
    df = load_data()
    print(f"Using full dataset: {len(df)} products.")

    df = apply_blocking(df)

    print("Loading CLIP model...")
    model = SentenceTransformer("clip-ViT-B-32")

    all_clusters = []
    id_offset = 0
    groups = list(df.groupby(["category", "price_bucket"]))

    for (cat, bucket), block in groups:
        block = block.reset_index(drop=True)
        print(f"\nBlock [{cat} / {bucket}]: {len(block)} items — embedding...")
        try:
            embeddings, valid_idx = generate_embeddings(block, model)
            if not embeddings:
                print(f"  No embeddings — skipping.")
                continue
            clusters = cluster_by_similarity(embeddings, valid_idx, block)
            block_path = _save_block(clusters, cat, bucket, id_offset)
            id_offset += len(clusters)
            all_clusters.extend(clusters)
            multi = sum(1 for c in clusters if len({i["store"] for i in c}) > 1)
            print(f"  → {len(clusters)} clusters formed ({multi} multi-store). Saved to {block_path}")
        except Exception as e:
            print(f"  ERROR in block [{cat}/{bucket}]: {e} — skipping.")
            continue

    save_embedding_cache()

    full_output_path = os.path.abspath(os.path.join(_OUTPUTS, "image_clusters_full.json"))
    save_clusters(all_clusters, full_output_path)

    # Also update the default clusters file used by cluster_insights.py
    default_output_path = os.path.abspath(os.path.join(_OUTPUTS, "image_clusters.json"))
    save_clusters(all_clusters, default_output_path)

    multi_store = [c for c in all_clusters if len(set(i["store"] for i in c)) > 1]
    elapsed = round(time.time() - start, 1)
    sizes = [len(c) for c in all_clusters]
    avg_size = round(sum(sizes) / len(sizes), 1) if sizes else 0
    max_size = max(sizes) if sizes else 0

    print(f"\nTotal clusters:            {len(all_clusters)}")
    print(f"Multi-store clusters:      {len(multi_store)}")
    print(f"Largest cluster:           {max_size} items")
    print(f"Average cluster size:      {avg_size} items")
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

    print(f"Full results saved to {full_output_path}")


if __name__ == "__main__":
    run()
