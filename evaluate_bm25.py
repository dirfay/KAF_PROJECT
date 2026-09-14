from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bm25_index import search


BASE_DIR = Path(__file__).resolve().parent

EVAL_PATH = (
    BASE_DIR
    / "data"
    / "retrieval_eval.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "bm25_evaluation.json"
)

TOP_K = 10


def load_queries() -> list[dict[str, Any]]:
    with EVAL_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def reciprocal_rank(
    ranked_sources: list[str],
    relevant_sources: set[str],
) -> float:
    for rank, source_id in enumerate(
        ranked_sources,
        start=1,
    ):
        if source_id in relevant_sources:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    ranked_sources: list[str],
    relevant_sources: set[str],
    k: int,
) -> float:
    retrieved = set(
        ranked_sources[:k]
    )

    if not relevant_sources:
        return 0.0

    return len(
        retrieved & relevant_sources
    ) / len(relevant_sources)


def dcg_at_k(
    ranked_sources: list[str],
    relevant_sources: set[str],
    k: int,
) -> float:
    score = 0.0

    for rank, source_id in enumerate(
        ranked_sources[:k],
        start=1,
    ):
        relevance = (
            1.0
            if source_id in relevant_sources
            else 0.0
        )

        if relevance == 0:
            continue

        score += relevance / math.log2(
            rank + 1
        )

    return score


def ndcg_at_k(
    ranked_sources: list[str],
    relevant_sources: set[str],
    k: int,
) -> float:
    actual = dcg_at_k(
        ranked_sources,
        relevant_sources,
        k,
    )

    ideal_relevances = min(
        len(relevant_sources),
        k,
    )

    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevances + 1,
        )
    )

    if ideal == 0:
        return 0.0

    return actual / ideal


def main() -> None:
    queries = load_queries()

    results = []

    recall_values = []
    mrr_values = []
    ndcg_values = []

    for item in queries:
        query_id = item["id"]
        query = item["query"]

        relevant_sources = set(
            item["relevant_sources"]
        )

        retrieved = search(
            query,
            top_k=TOP_K,
        )

        ranked_sources = [
            result["source_id"]
            for result in retrieved
        ]

        # Прибираємо повтори source_id,
        # зберігаючи першу появу.
        unique_ranked_sources = []

        seen = set()

        for source_id in ranked_sources:
            if source_id in seen:
                continue

            seen.add(source_id)
            unique_ranked_sources.append(
                source_id
            )

        ranked_sources = (
            unique_ranked_sources
        )

        r1 = recall_at_k(
            ranked_sources,
            relevant_sources,
            1,
        )

        r3 = recall_at_k(
            ranked_sources,
            relevant_sources,
            3,
        )

        r5 = recall_at_k(
            ranked_sources,
            relevant_sources,
            5,
        )

        r10 = recall_at_k(
            ranked_sources,
            relevant_sources,
            10,
        )

        rr = reciprocal_rank(
            ranked_sources,
            relevant_sources,
        )

        ndcg5 = ndcg_at_k(
            ranked_sources,
            relevant_sources,
            5,
        )

        ndcg10 = ndcg_at_k(
            ranked_sources,
            relevant_sources,
            10,
        )

        record = {
            "id": query_id,
            "query": query,
            "relevant_sources": sorted(
                relevant_sources
            ),
            "ranked_sources": ranked_sources,
            "recall_at_1": r1,
            "recall_at_3": r3,
            "recall_at_5": r5,
            "recall_at_10": r10,
            "reciprocal_rank": rr,
            "ndcg_at_5": ndcg5,
            "ndcg_at_10": ndcg10,
        }

        results.append(record)

        recall_values.append(
            r10
        )

        mrr_values.append(
            rr
        )

        ndcg_values.append(
            ndcg10
        )

    count = len(results)

    summary = {
        "system": "E1_BM25",
        "queries": count,
        "top_k": TOP_K,
        "recall_at_10": (
            sum(recall_values) / count
            if count
            else 0.0
        ),
        "mrr": (
            sum(mrr_values) / count
            if count
            else 0.0
        ),
        "ndcg_at_10": (
            sum(ndcg_values) / count
            if count
            else 0.0
        ),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 80)
    print("BM25 EVALUATION — E1")
    print("=" * 80)
    print()
    print(
        f"Запитів: {summary['queries']}"
    )
    print(
        f"Recall@10: "
        f"{summary['recall_at_10']:.4f}"
    )
    print(
        f"MRR:       "
        f"{summary['mrr']:.4f}"
    )
    print(
        f"nDCG@10:   "
        f"{summary['ndcg_at_10']:.4f}"
    )
    print()
    print(
        "Результати:",
        OUTPUT_PATH,
    )
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
