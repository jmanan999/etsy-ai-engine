import json
import os
import random
import re

_OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "outputs")
_INPUT = os.path.join(_OUTPUTS, "image_clusters_full.json")

_CATEGORY_KEYWORDS = {
    "ring":      ["ring", "band", "signet", "dome", "knuckle", "thumb"],
    "earring":   ["earring", "earrings", "hoop", "stud", "dangle", "drop", "huggie"],
    "bracelet":  ["bracelet", "bangle", "anklet", "ankle", "kada", "wristlet"],
    "necklace":  ["necklace", "pendant", "chain", "choker", "locket", "charm"],
    "cuff":      ["cuff", "armlet", "armband", "upper arm"],
}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "of", "in", "to", "with",
    "by", "on", "is", "it", "its", "be", "as", "at", "this", "that",
    "from", "are", "was", "but", "not", "so", "my", "your",
}


def _fix_url(url: str) -> str:
    return url.replace("il_75x75", "il_794x794")


def _get_category(title: str) -> str:
    t = title.lower()
    for cat, kws in _CATEGORY_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return cat
    return "other"


def _top_words(titles: list[str], n: int = 10) -> list[str]:
    counts: dict[str, int] = {}
    for title in titles:
        for word in re.findall(r"[a-z]{3,}", title.lower()):
            if word not in _STOPWORDS:
                counts[word] = counts.get(word, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda x: -x[1])[:n]]


def _detect_red_flags(items: list[dict]) -> list[str]:
    flags = []
    titles = [i["title"] for i in items]

    # Mixed categories
    cats = {_get_category(t) for t in titles} - {"other"}
    if len(cats) > 1:
        flags.append(f"Mixed product categories detected: {', '.join(sorted(cats))}")

    # Low title overlap across cluster
    if len(titles) >= 2:
        all_words = [set(re.findall(r"[a-z]{3,}", t.lower())) - _STOPWORDS for t in titles]
        overlaps = []
        for a in range(len(all_words)):
            for b in range(a + 1, len(all_words)):
                union = all_words[a] | all_words[b]
                inter = all_words[a] & all_words[b]
                overlaps.append(len(inter) / len(union) if union else 0)
        avg_overlap = sum(overlaps) / len(overlaps)
        if avg_overlap < 0.05:
            flags.append(f"Very low title word overlap across items (avg Jaccard: {avg_overlap:.2f})")

    return flags


def _print_cluster(cluster: dict) -> None:
    cid = cluster["cluster_id"]
    items = cluster["items"]
    stores = list({i["store"] for i in items})
    titles = [i["title"] for i in items]

    sims = [i["img_sim"] for i in items if "img_sim" in i]
    avg_sim = round(sum(sims) / len(sims), 3) if sims else None

    print("=" * 60)
    print(f"Cluster {cid}  |  Items: {len(items)}  |  Stores: {len(stores)}")
    print("=" * 60)

    for item in items:
        print(f"  Store:   {item['store']}")
        print(f"  Title:   {item['title'][:100]}")
        print(f"  Image:   {_fix_url(item['image_url'])}")
        if "img_sim" in item:
            print(f"  img_sim: {item['img_sim']}  |  title_sim: {item['title_sim']}  |  condition: {item['condition']}")
        print()

    top_words = _top_words(titles)
    print(f"  Top words:    {', '.join(top_words)}")
    if avg_sim is not None:
        print(f"  Avg img_sim:  {avg_sim}")

    flags = _detect_red_flags(items)
    if flags:
        print()
        print("  ⚠️  POSSIBLE BAD CLUSTER")
        for f in flags:
            print(f"     → {f}")

    print()


def run():
    if not os.path.exists(_INPUT):
        print(f"Input file not found: {_INPUT}")
        print("Run image_clustering.py first to generate the full dataset clusters.")
        return

    with open(_INPUT) as f:
        data = json.load(f)

    all_clusters = data["clusters"]
    total_items = sum(len(c["items"]) for c in all_clusters)
    multi_store = [c for c in all_clusters if len({i["store"] for i in c["items"]}) >= 2]

    print("=" * 60)
    print("CLUSTER VALIDATION REPORT")
    print("=" * 60)
    print(f"  Total clusters:        {len(all_clusters)}")
    print(f"  Total items:           {total_items}")
    print(f"  Multi-store clusters:  {len(multi_store)}")
    print()

    if not multi_store:
        print("No multi-store clusters found. Nothing to inspect.")
        return

    sorted_clusters = sorted(multi_store, key=lambda c: len(c["items"]), reverse=True)

    # --- Top 5 largest ---
    print("=" * 60)
    print("TOP 5 LARGEST CLUSTERS")
    print("=" * 60)
    print()
    for c in sorted_clusters[:5]:
        _print_cluster(c)

    # --- 5 random ---
    random.seed(42)
    random_sample = random.sample(multi_store, min(5, len(multi_store)))
    print("=" * 60)
    print("5 RANDOM CLUSTERS")
    print("=" * 60)
    print()
    for c in random_sample:
        _print_cluster(c)

    # --- Exactly 2 stores (edge case) ---
    two_store = [c for c in multi_store if len({i["store"] for i in c["items"]}) == 2]
    edge_sample = two_store[:3]
    print("=" * 60)
    print("EDGE CASE: CLUSTERS WITH EXACTLY 2 STORES")
    print("=" * 60)
    print()
    for c in edge_sample:
        _print_cluster(c)

    total_flagged = sum(1 for c in multi_store if _detect_red_flags(c["items"]))
    print("=" * 60)
    print(f"Red-flagged clusters (across all multi-store): {total_flagged} / {len(multi_store)}")
    print("=" * 60)


if __name__ == "__main__":
    run()
