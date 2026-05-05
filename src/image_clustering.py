import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

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
_BATCH_SIZE = 32
_DOWNLOAD_WORKERS = 30
_BLOCK_SPLIT_THRESHOLD = 2000

_CATEGORY_KEYWORDS = {
    "ring":      ["ring", "band", "signet", "dome", "knuckle", "thumb"],
    "earring":   ["earring", "earrings", "hoop", "stud", "dangle", "drop", "huggie"],
    "bracelet":  ["bracelet", "bangle", "anklet", "ankle", "kada", "wristlet"],
    "necklace":  ["necklace", "pendant", "chain", "choker", "locket", "charm"],
    "cuff":      ["cuff", "armlet", "arm band", "armband", "upper arm"],
    "other":     [],
}

# Sub-keywords used to split oversized blocks
_SUBCATEGORY_KEYWORDS = {
    "ring":     [["signet", "dome", "thumb", "knuckle"], ["band", "plain", "simple"], ["stone", "gem", "crystal"]],
    "earring":  [["hoop", "huggie"], ["stud", "flat"], ["dangle", "drop"]],
    "bracelet": [["bangle", "kada"], ["anklet", "ankle"], ["chain", "link"]],
    "necklace": [["pendant", "charm", "locket"], ["chain", "link"], ["choker"]],
    "cuff":     [["cuff", "armlet"], ["armband", "upper arm"]],
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


def _split_large_block(block: pd.DataFrame, cat: str) -> list[pd.DataFrame]:
    """Split an oversized block using sub-category keywords."""
    sub_groups = _SUBCATEGORY_KEYWORDS.get(cat, [])
    if not sub_groups:
        return [block]

    assigned = pd.Series(False, index=block.index)
    splits = []
    for kws in sub_groups:
        mask = block["title"].str.lower().apply(lambda t: any(kw in t for kw in kws)) & ~assigned
        sub = block[mask]
        if len(sub) > 0:
            splits.append(sub.reset_index(drop=True))
            assigned |= mask

    remainder = block[~assigned]
    if len(remainder) > 0:
        splits.append(remainder.reset_index(drop=True))

    return splits if splits else [block]


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


def _fetch_image(url: str) -> tuple[str, Image.Image | None]:
    """Download one image; return (url, PIL Image or None)."""
    hires = url.replace("il_75x75", "il_794x794")
    try:
        r = requests.get(hires, timeout=5)
        r.raise_for_status()
        return url, Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return url, None


def generate_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> tuple[list, list[int]]:
    """
    Stage 1: parallel download of unique URLs not in cache.
    Stage 2: batch encode all new images at once.
    Maps embeddings back to every row sharing the same URL.
    """
    urls = df["image_url"].tolist()
    unique_urls = list(dict.fromkeys(urls))  # preserve order, deduplicate

    # Split into cached vs needs download
    cached_urls = {u for u in unique_urls if u in _embedding_cache}
    to_download = [u for u in unique_urls if u not in cached_urls]

    t_dl = time.time()
    url_to_image: dict[str, Image.Image] = {}
    if to_download:
        with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as ex:
            for url, img in ex.map(_fetch_image, to_download):
                if img is not None:
                    url_to_image[url] = img
    dl_elapsed = round(time.time() - t_dl, 1)

    # Stage 2: batch embed all new images
    t_emb = time.time()
    new_urls = list(url_to_image.keys())
    new_images = [url_to_image[u] for u in new_urls]
    if new_images:
        raw = model.encode(new_images, batch_size=_BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        normalized = raw / (norms + 1e-9)
        for url, emb in zip(new_urls, normalized):
            _embedding_cache[url] = emb
    emb_elapsed = round(time.time() - t_emb, 1)

    cache_hits = len(cached_urls)
    new_count = len(new_images)
    failed = len(to_download) - new_count
    print(f"  Unique URLs: {len(unique_urls)} | cache hits: {cache_hits} | "
          f"new: {new_count} | failed: {failed} | "
          f"download: {dl_elapsed}s | embed: {emb_elapsed}s")

    # Map embeddings back to all rows
    embeddings, valid_idx = [], []
    for idx, url in enumerate(urls):
        emb = _embedding_cache.get(url)
        if emb is not None:
            embeddings.append(emb)
            valid_idx.append(idx)

    return embeddings, valid_idx


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Embeddings are pre-normalized — dot product is sufficient
    return float(np.dot(a, b))


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
    # Early rejection: skip cosine if titles share zero words
    if not (set(anchor_title.lower().split()) & set(title_j.lower().split())):
        anchor_img_sim = _cosine_similarity(anchor_emb, emb_j)
        if anchor_img_sim < 0.80:
            return False, anchor_img_sim, 0.0, ""

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
    cluster_embs_list = []
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


def _block_output_path(cat: str, bucket: str, sub: int | None = None) -> str:
    os.makedirs(_CLUSTERS_DIR, exist_ok=True)
    suffix = f"_{sub}" if sub is not None else ""
    return os.path.join(_CLUSTERS_DIR, f"clusters_{cat}_{bucket}{suffix}.json")


def _save_block(clusters: list[list], cat: str, bucket: str, id_offset: int, sub: int | None = None) -> str:
    path = _block_output_path(cat, bucket, sub)
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
            {"cluster_id": f"P{str(i+1).zfill(4)}", "items": items}
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

        # Split oversized blocks into sub-blocks
        if len(block) > _BLOCK_SPLIT_THRESHOLD:
            sub_blocks = _split_large_block(block, cat)
            print(f"\nBlock [{cat} / {bucket}]: {len(block)} items → split into {len(sub_blocks)} sub-blocks")
        else:
            sub_blocks = [block]

        for sub_idx, sub_block in enumerate(sub_blocks):
            sub_label = f"sub-{sub_idx}" if len(sub_blocks) > 1 else None
            label = f"[{cat} / {bucket}{f' / {sub_label}' if sub_label else ''}]"
            print(f"\n  Block {label}: {len(sub_block)} items — downloading + embedding...")
            try:
                embeddings, valid_idx = generate_embeddings(sub_block, model)
                if not embeddings:
                    print(f"  No embeddings — skipping.")
                    continue
                clusters = cluster_by_similarity(embeddings, valid_idx, sub_block)
                block_path = _save_block(clusters, cat, bucket, id_offset, sub=sub_idx if len(sub_blocks) > 1 else None)
                id_offset += len(clusters)
                all_clusters.extend(clusters)
                multi = sum(1 for c in clusters if len({i["store"] for i in c}) > 1)
                print(f"  → {len(clusters)} clusters ({multi} multi-store). Saved to {block_path}")
            except Exception as e:
                print(f"  ERROR in block {label}: {e} — skipping.")
                continue

    save_embedding_cache()

    full_output_path = os.path.abspath(os.path.join(_OUTPUTS, "image_clusters_full.json"))
    save_clusters(all_clusters, full_output_path)

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


# --- OPTIONAL (DISABLED): FAISS-based nearest neighbor search ---
# Uncomment to enable approximate nearest neighbor retrieval per block.
# Requires: pip install faiss-cpu
#
# import faiss
# def build_faiss_index(embeddings: list[np.ndarray]) -> faiss.IndexFlatIP:
#     matrix = np.stack(embeddings).astype("float32")
#     index = faiss.IndexFlatIP(matrix.shape[1])  # inner product = cosine on normalized vectors
#     index.add(matrix)
#     return index
#
# def faiss_candidates(index, query_emb: np.ndarray, k: int = 50) -> list[int]:
#     q = query_emb.astype("float32").reshape(1, -1)
#     _, indices = index.search(q, k)
#     return indices[0].tolist()


if __name__ == "__main__":
    run()
