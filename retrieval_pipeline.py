from __future__ import annotations

import time
from typing import Any

from reranker import reranked_hybrid_search


def retrieve(
    query: str,
    candidates: int = 20,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Єдина точка входу для retrieval-рівня.

    Pipeline:

        query
          ↓
        BM25
          +
        Semantic
          ↓
        RRF
          ↓
        Reranker
          ↓
        TOP-K
    """

    query = query.strip()

    if not query:
        return {
            "query": query,
            "results": [],
            "latency_seconds": 0.0,
        }

    started = time.perf_counter()

    results = reranked_hybrid_search(
        query,
        candidates=candidates,
        top_k=top_k,
    )

    latency = (
        time.perf_counter()
        - started
    )

    normalized_results = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        record = result["record"]

        normalized_results.append(
            {
                "rank": rank,
                "id": record.get("id"),
                "source_id": record.get(
                    "source_id"
                ),
                "title": record.get(
                    "title"
                ),
                "section": record.get(
                    "section"
                ),
                "page": record.get(
                    "page"
                ),
                "text": record.get(
                    "text",
                    "",
                ),
                "metadata": record.get(
                    "metadata",
                    {},
                ),
                "reranker_score": result.get(
                    "reranker_score",
                    0.0,
                ),
                "rrf_score": result.get(
                    "rrf_score",
                    0.0,
                ),
                "bm25_rank": result.get(
                    "bm25_rank"
                ),
                "semantic_rank": result.get(
                    "semantic_rank"
                ),
                "bm25_score": result.get(
                    "bm25_score"
                ),
                "semantic_score": result.get(
                    "semantic_score"
                ),
            }
        )

    return {
        "query": query,
        "results": normalized_results,
        "latency_seconds": latency,
    }


def build_context(
    retrieval_result: dict[str, Any],
    max_chunks: int = 5,
) -> str:
    """
    Формує текстовий контекст для майбутнього RAG.
    """

    chunks = retrieval_result.get(
        "results",
        [],
    )

    parts = []

    for item in chunks[:max_chunks]:
        source_id = item.get(
            "source_id",
            "",
        )

        title = item.get(
            "title",
            "",
        )

        section = item.get(
            "section",
            "",
        )

        page = item.get(
            "page"
        )

        text = item.get(
            "text",
            "",
        )

        header_parts = []

        if source_id:
            header_parts.append(
                f"source={source_id}"
            )

        if title:
            header_parts.append(
                f"title={title}"
            )

        if section:
            header_parts.append(
                f"section={section}"
            )

        if page:
            header_parts.append(
                f"page={page}"
            )

        header = " | ".join(
            header_parts
        )

        parts.append(
            f"[{header}]\n{text}"
        )

    return "\n\n".join(parts)


def format_sources(
    retrieval_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Готує компактний список джерел
    для повернення через API.
    """

    sources = []

    seen = set()

    for item in retrieval_result.get(
        "results",
        [],
    ):
        source_id = item.get(
            "source_id"
        )

        if not source_id:
            continue

        if source_id in seen:
            continue

        seen.add(source_id)

        sources.append(
            {
                "source_id": source_id,
                "title": item.get(
                    "title"
                ),
                "section": item.get(
                    "section"
                ),
                "page": item.get(
                    "page"
                ),
                "score": item.get(
                    "reranker_score",
                    0.0,
                ),
            }
        )

    return sources
