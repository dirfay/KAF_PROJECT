from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from reranker import reranked_hybrid_search


BASE_DIR = Path(__file__).resolve().parent

EVAL_PATH = (
    BASE_DIR
    / "data"
    / "retrieval_eval.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "reranker_evaluation.json"
)


def load_eval() -> list[dict[str, Any]]:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено файл:\n{EVAL_PATH}"
        )

    with EVAL_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise TypeError(
            "retrieval_eval.json має містити список."
        )

    return data


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
        if source_id in relevant_sources:
            score += (
                1.0
                / math.log2(rank + 1)
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

    ideal_count = min(
        len(relevant_sources),
        k,
    )

    if ideal_count == 0:
        return 0.0

    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_count + 1,
        )
    )

    if ideal == 0.0:
        return 0.0

    return actual / ideal


def main() -> None:
    evaluation = load_eval()

    print("=" * 80)
    print("RERANKER RETRIEVAL EVALUATION — E4")
    print("=" * 80)

    print()
    print(
        f"Запитів: {len(evaluation)}"
    )

    print()
    print(
        "Метод: Hybrid BM25 + Semantic + RRF + Reranker"
    )

    print(
        "Hybrid candidates: 20"
    )

    print(
        "Final TOP-K: 10"
    )

    results = []

    total_recall = 0.0
    total_rr = 0.0
    total_ndcg = 0.0
    total_latency = 0.0

    for item in evaluation:
        query_id = item["id"]
        query = item["query"]

        relevant_sources = set(
            item["relevant_sources"]
        )

        started = time.perf_counter()

        search_results = reranked_hybrid_search(
            query,
            candidates=20,
            top_k=10,
        )

        latency = (
            time.perf_counter()
            - started
        )

        ranked_sources: list[str] = []

        for result in search_results:
            source_id = result["record"].get(
                "source_id"
            )

            if (
                source_id
                and source_id not in ranked_sources
            ):
                ranked_sources.append(
                    source_id
                )

        top10 = ranked_sources[:10]

        hits = sum(
            1
            for source_id in top10
            if source_id in relevant_sources
        )

        recall = (
            hits / len(relevant_sources)
            if relevant_sources
            else 0.0
        )

        rr = reciprocal_rank(
            top10,
            relevant_sources,
        )

        ndcg = ndcg_at_k(
            top10,
            relevant_sources,
            10,
        )

        total_recall += recall
        total_rr += rr
        total_ndcg += ndcg
        total_latency += latency

        results.append(
            {
                "id": query_id,
                "query": query,
                "relevant_sources": sorted(
                    relevant_sources
                ),
                "ranked_sources": ranked_sources,
                "top10_sources": top10,
                "recall_at_10": recall,
                "reciprocal_rank": rr,
                "ndcg_at_10": ndcg,
                "latency_seconds": latency,
            }
        )

        print(
            f"[{query_id}] "
            f"Recall={recall:.4f} "
            f"RR={rr:.4f} "
            f"nDCG={ndcg:.4f} "
            f"Latency={latency:.3f}s"
        )

    count = len(results)

    summary = {
        "version": "E4",
        "method": (
            "Hybrid BM25 + Semantic + RRF + Reranker"
        ),
        "semantic_model": (
            "intfloat/multilingual-e5-small"
        ),
        "reranker_model": (
            "BAAI/bge-reranker-v2-m3"
        ),
        "rrf_k": 60,
        "candidates_per_method": 20,
        "reranker_candidates": 20,
        "final_top_k": 10,
        "queries": count,
        "recall_at_10": (
            total_recall / count
            if count
            else 0.0
        ),
        "mrr": (
            total_rr / count
            if count
            else 0.0
        ),
        "ndcg_at_10": (
            total_ndcg / count
            if count
            else 0.0
        ),
        "mean_latency_seconds": (
            total_latency / count
            if count
            else 0.0
        ),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("E4 — ПІДСУМОК")
    print("=" * 80)

    print()
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

    print(
        f"Latency:   "
        f"{summary['mean_latency_seconds']:.4f} s"
    )

    print()
    print(
        f"Результати: {OUTPUT_PATH}"
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
