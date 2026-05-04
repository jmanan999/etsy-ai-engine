import json
import os

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

_OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "outputs")
_SIMILARITY_THRESHOLD = 0.85
_TOP_N = 500


def load_data() -> pd.DataFrame:
    cleaned = pd.read_csv(os.path.join(_OUTPUTS, "cleaned_data.csv"),
                          dtype={"transaction_id": str})
    images = pd.read_csv(os.path.join(_OUTPUTS, "image_urls.csv"),
                         dtype={"Transaction ID": str})
    images = images.rename(columns={"Transaction ID": "transaction_id"})

    df = cleaned.merge(images, on="transaction_id", how="inner")
    df = df.dropna(subset=["image_url", "title", "store_name"])

    # One image per unique title — pick the first occurrence
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    print(f"Unique titled products with images: {len(df)}")
    return df


def _load_image(url: str):
    try:
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


def _final_score(img_sim: float, title_sim: float) -> float:
    return 0.7 * img_sim + 0.3 * title_sim


def cluster_by_similarity(embeddings: list, valid_idx: list, df: pd.DataFrame) -> list[dict]:
    """Greedy clustering using combined image + title similarity."""
    clusters = []
    assigned = set()

    for i, (emb_i, idx_i) in enumerate(zip(embeddings, valid_idx)):
        if i in assigned:
            continue
        title_i = df.at[idx_i, "title"]
        cluster_items = [{
            "title": title_i,
            "store": df.at[idx_i, "store_name"],
            "image_url": df.at[idx_i, "image_url"],
        }]
        assigned.add(i)
        for j, (emb_j, idx_j) in enumerate(zip(embeddings, valid_idx)):
            if j in assigned:
                continue
            title_j = df.at[idx_j, "title"]
            img_sim = _cosine_similarity(emb_i, emb_j)
            title_sim = compute_title_similarity(title_i, title_j)
            score = _final_score(img_sim, title_sim)
            if score >= _SIMILARITY_THRESHOLD:
                cluster_items.append({
                    "title": title_j,
                    "store": df.at[idx_j, "store_name"],
                    "image_url": df.at[idx_j, "image_url"],
                    "similarity_score": round(score, 3),
                })
                assigned.add(j)
        clusters.append(cluster_items)

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
    print("Loading data...")
    df = load_data()
    sample = df.head(_TOP_N).reset_index(drop=True)
    print(f"Using top {len(sample)} products for clustering.")

    print("\nLoading CLIP model...")
    model = SentenceTransformer("clip-ViT-B-32")

    print("\nFetching images and generating embeddings...")
    embeddings, valid_idx = generate_embeddings(sample, model)
    print(f"Successfully embedded {len(embeddings)} images.")

    print("\nClustering...")
    clusters = cluster_by_similarity(embeddings, valid_idx, sample)

    output_path = os.path.abspath(os.path.join(_OUTPUTS, "image_clusters.json"))
    save_clusters(clusters, output_path)

    multi_store = [c for c in clusters if len(set(i["store"] for i in c)) > 1]
    print(f"\nTotal clusters: {len(clusters)}")
    print(f"Clusters with multiple stores: {len(multi_store)}")

    if multi_store:
        print("\nSample multi-store cluster:")
        for item in multi_store[0][:5]:
            score_str = f"  score={item['similarity_score']}" if "similarity_score" in item else ""
            print(f"  Store:  {item['store']}")
            print(f"  Title:  {item['title'][:90]}")
            print(f"  Image:  {item['image_url']}{score_str}")
            print()

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    run()
