from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from bm25_index import search as bm25_search
from embeddings import semantic_search


BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_PATH = (
    BASE_DIR
    / "data"
    / "knowledge_documents.jsonl"
)

RRF_K = 60

DEFAULT_CANDIDATES = 20

DEFAULT_TOP_K = 10


def load_knowledge() -> list[dict[str, Any]]:
    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено базу знань:\n{KNOWLEDGE_PATH}"
        )

    records: list[dict[str, Any]] = []

    with KNOWLEDGE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Помилка JSON у рядку "
                    f"{line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def normalize_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Приводить результати BM25 та Semantic
    до єдиного формату:

    {
        "record": {...},
        "score": float
    }
    """

    if "record" in result:
        record = result["record"]

        return {
            "record": record,
            "score": float(
                result.get(
                    "score",
                    0.0,
                )
            ),
        }

    record = dict(result)

    score = record.pop(
        "score",
        0.0,
    )

    return {
        "record": record,
        "score": float(score),
    }


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    def add_result(
        result: dict[str, Any],
        rank: int,
        method: str,
    ) -> None:
        normalized = normalize_result(
            result
        )

        record = normalized["record"]

        record_id = str(
            record.get(
                "id",
                "",
            )
        )

        if not record_id:
            return

        if record_id not in fused:
            fused[record_id] = {
                "record": record,
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
                "bm25_score": None,
                "semantic_score": None,
            }

        item = fused[record_id]

        item["rrf_score"] += (
            1.0
            / (RRF_K + rank)
        )

        if method == "bm25":
            item["bm25_rank"] = rank
            item["bm25_score"] = normalized[
                "score"
            ]

        elif method == "semantic":
            item["semantic_rank"] = rank
            item["semantic_score"] = normalized[
                "score"
            ]

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        add_result(
            result,
            rank,
            "bm25",
        )

    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):
        add_result(
            result,
            rank,
            "semantic",
        )

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            item["rrf_score"],
            item["bm25_rank"] is not None,
            item["semantic_rank"] is not None,
        ),
        reverse=True,
    )

    return ranked[:top_k]


def hybrid_search(
    query: str,
    candidates: int = DEFAULT_CANDIDATES,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    query = query.strip()

    if not query:
        return []

    bm25_results = bm25_search(
        query,
        top_k=candidates,
    )

    semantic_results = semantic_search(
        query,
        top_k=candidates,
    )

    fused_results = reciprocal_rank_fusion(
        bm25_results,
        semantic_results,
        top_k=top_k,
    )

    return fused_results


def print_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 80)
    print("HYBRID SEARCH — E3")
    print("=" * 80)

    print()
    print("Запит:")
    print(query)

    print()
    print(
        f"RRF k: {RRF_K}"
    )

    print(
        f"Кандидатів від кожного методу: "
        f"{DEFAULT_CANDIDATES}"
    )

    print()
    print("-" * 80)

    if not results:
        print(
            "Результатів не знайдено."
        )
        return

    for position, item in enumerate(
        results,
        start=1,
    ):
        record = item["record"]

        print(
            f"#{position} "
            f"RRF={item['rrf_score']:.6f}"
        )

        print(
            "ID:",
            record.get("id"),
        )

        print(
            "SOURCE:",
            record.get("source_id"),
        )

        print(
            "TITLE:",
            record.get("title"),
        )

        if record.get("section"):
            print(
                "SECTION:",
                record.get("section"),
            )

        if record.get("page"):
            print(
                "PAGE:",
                record.get("page"),
            )

        print(
            "BM25 rank:",
            item["bm25_rank"],
        )

        print(
            "Semantic rank:",
            item["semantic_rank"],
        )

        if item["bm25_score"] is not None:
            print(
                f"BM25 score: "
                f"{item['bm25_score']:.6f}"
            )

        if item["semantic_score"] is not None:
            print(
                f"Semantic score: "
                f"{item['semantic_score']:.6f}"
            )

        print()
        print(
            str(
                record.get(
                    "text",
                    "",
                )
            )[:1200]
        )

        print("-" * 80)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Використання:"
        )
        print(
            '  python hybrid_search.py "текст"'
        )
        return

    query = " ".join(
        sys.argv[1:]
    )

    results = hybrid_search(
        query,
        candidates=DEFAULT_CANDIDATES,
        top_k=DEFAULT_TOP_K,
    )

    print_results(
        query,
        results,
    )


if __name__ == "__main__":
    main()
