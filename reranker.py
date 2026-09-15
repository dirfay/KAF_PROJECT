from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder

from hybrid_search import hybrid_search


BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DEFAULT_CANDIDATES = 20

DEFAULT_TOP_K = 10


_RERANKER: CrossEncoder | None = None


def get_reranker(
    show_message: bool = True,
) -> CrossEncoder:
    global _RERANKER

    if _RERANKER is not None:
        return _RERANKER

    if show_message:
        print()
        print("=" * 80)
        print("RERANKER MODEL — LOAD")
        print("=" * 80)

        print()
        print("Модель:")
        print(MODEL_NAME)

        print()
        print("Завантаження reranker-моделі...")

    started = time.perf_counter()

    _RERANKER = CrossEncoder(
        MODEL_NAME
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if show_message:
        print(
            f"Reranker завантажений "
            f"за {elapsed:.2f} с"
        )

    return _RERANKER


def build_passage_text(
    record: dict[str, Any],
) -> str:
    title = str(
        record.get(
            "title",
            "",
        )
    )

    section = str(
        record.get(
            "section",
            "",
        )
    )

    text = str(
        record.get(
            "text",
            "",
        )
    )

    parts: list[str] = []

    if title:
        parts.append(title)

    if section:
        parts.append(section)

    if text:
        parts.append(text)

    return "\n".join(parts).strip()


def rerank_results(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    if not candidates:
        return []

    reranker = get_reranker()

    pairs = []

    for candidate in candidates:
        record = candidate["record"]

        passage = build_passage_text(
            record
        )

        pairs.append(
            [
                query,
                passage,
            ]
        )

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    results: list[dict[str, Any]] = []

    for candidate, score in zip(
        candidates,
        scores,
    ):
        results.append(
            {
                "record": candidate["record"],
                "reranker_score": float(score),
                "rrf_score": float(
                    candidate.get(
                        "rrf_score",
                        0.0,
                    )
                ),
                "bm25_rank": candidate.get(
                    "bm25_rank"
                ),
                "semantic_rank": candidate.get(
                    "semantic_rank"
                ),
                "bm25_score": candidate.get(
                    "bm25_score"
                ),
                "semantic_score": candidate.get(
                    "semantic_score"
                ),
            }
        )

    results.sort(
        key=lambda item: item[
            "reranker_score"
        ],
        reverse=True,
    )

    return results[:top_k]


def reranked_hybrid_search(
    query: str,
    candidates: int = DEFAULT_CANDIDATES,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    hybrid_results = hybrid_search(
        query,
        candidates=candidates,
        top_k=candidates,
    )

    return rerank_results(
        query,
        hybrid_results,
        top_k=top_k,
    )


def print_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 80)
    print("RERANKED HYBRID SEARCH — E4")
    print("=" * 80)

    print()
    print("Запит:")
    print(query)

    print()
    print(
        f"Hybrid candidates: "
        f"{DEFAULT_CANDIDATES}"
    )

    print(
        f"Final TOP-K: "
        f"{DEFAULT_TOP_K}"
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
            f"reranker="
            f"{item['reranker_score']:.6f}"
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
            "RRF:",
            f"{item['rrf_score']:.6f}",
        )

        print(
            "BM25 rank:",
            item["bm25_rank"],
        )

        print(
            "Semantic rank:",
            item["semantic_rank"],
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
            '  python reranker.py "текст"'
        )
        return

    query = " ".join(
        sys.argv[1:]
    )

    results = reranked_hybrid_search(
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
