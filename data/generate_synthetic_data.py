import random
from typing import Dict, List

import numpy as np
import pandas as pd


def build_item_table(
    product_phrases: List[str],
    adjectives: List[str],
    materials: List[str],
    extras: List[str],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    item_ids = np.arange(200)
    item_phrase = rng.choice(product_phrases, size=len(item_ids))
    item_adj = rng.choice(adjectives, size=len(item_ids))
    item_material = rng.choice(materials, size=len(item_ids))
    item_extra = rng.choice(extras, size=len(item_ids))

    descriptions = [
        f"{adj} {material} {phrase} with {extra}"
        for adj, material, phrase, extra in zip(item_adj, item_material, item_phrase, item_extra)
    ]

    return pd.DataFrame({"item_id": item_ids, "item_description": descriptions, "phrase": item_phrase})


def build_phrase_lookup(item_table: pd.DataFrame, product_phrases: List[str]) -> Dict[str, List[str]]:
    phrase_to_items: Dict[str, List[str]] = {}
    for phrase in product_phrases:
        matching = item_table.loc[item_table["phrase"] == phrase, "item_description"].tolist()
        phrase_to_items[phrase] = matching
    return phrase_to_items


def build_pairs(
    product_phrases: List[str],
    phrase_to_items: Dict[str, List[str]],
    num_pairs: int = 1000,
    min_pairs_per_phrase: int = 5,
) -> pd.DataFrame:
    rows = []

    # Guarantee each query phrase gets multiple positive/negative examples.
    for phrase in product_phrases:
        for _ in range(min_pairs_per_phrase):
            pos_item = random.choice(phrase_to_items[phrase])
            rows.append({"query_text": phrase, "item_description": pos_item, "label": 1})

            other_phrases = [p for p in product_phrases if p != phrase]
            neg_phrase = random.choice(other_phrases)
            neg_item = random.choice(phrase_to_items[neg_phrase])
            rows.append({"query_text": phrase, "item_description": neg_item, "label": 0})

    remaining_pairs = max(num_pairs - len(product_phrases) * min_pairs_per_phrase, 0)

    # Fill the rest with random sampling to reach the requested dataset size.
    for _ in range(remaining_pairs):
        query = random.choice(product_phrases)

        pos_item = random.choice(phrase_to_items[query])
        rows.append({"query_text": query, "item_description": pos_item, "label": 1})

        other_phrases = [p for p in product_phrases if p != query]
        neg_phrase = random.choice(other_phrases)
        neg_item = random.choice(phrase_to_items[neg_phrase])
        rows.append({"query_text": query, "item_description": neg_item, "label": 0})

    return pd.DataFrame(rows)


def main() -> None:
    random.seed(42)

    product_phrases = [
        "running shoes",
        "hiking boots",
        "yoga mat",
        "coffee mug",
        "water bottle",
        "wireless headphones",
        "bluetooth speaker",
        "phone case",
        "laptop stand",
        "desk lamp",
    ]

    adjectives = [
        "red",
        "blue",
        "green",
        "black",
        "white",
        "insulated",
        "lightweight",
        "ergonomic",
        "compact",
        "wireless",
        "portable",
        "sleek",
        "foldable",
    ]

    materials = [
        "mesh",
        "leather",
        "stainless steel",
        "ceramic",
        "plastic",
        "bamboo",
        "aluminum",
        "silicone",
    ]

    extras = [
        "arch support",
        "non-slip grip",
        "double-wall insulation",
        "noise cancelling",
        "spill-proof lid",
        "USB-C charging",
        "adjustable height",
        "foldable design",
        "LED dimmer",
        "shock absorption",
        "quick-dry lining",
        "magnetic closure",
    ]

    item_table = build_item_table(product_phrases, adjectives, materials, extras)
    phrase_to_items = build_phrase_lookup(item_table, product_phrases)
    df = build_pairs(product_phrases, phrase_to_items, num_pairs=1000, min_pairs_per_phrase=5)

    print("Item catalog sample:")
    print(item_table.head(), "\n")

    print("Pair dataset sample:")
    print(df.head(), "\n")

    print("Label counts:")
    print(df["label"].value_counts())

    output_path = "data/synthetic_query_item_pairs.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
