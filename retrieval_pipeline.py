from __future__ import annotations

import time
from typing import Any

from hybrid_search import hybrid_search


def retrieve(
    query: str,
    candidates: int = 20,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Робочий retrieval pipeline E3.

    query
      ↓
    BM25 + Semantic
      ↓
    RRF
      ↓
    TOP-K
    """

    query = (query or "").strip()

    if not query:
        return {
            "query": query,
            "results": [],
            "latency_seconds": 0.0,
            "method": "hybrid_rrf",
        }

    started = time.perf_counter()

    results = hybrid_search(
        query,
        candidates=candidates,
        top_k=top_k,
    )

    latency = time.perf_counter() - started

    normalized_results = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        record = result.get(
            "record",
            {},
        )

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
                "url": record.get(
                    "url"
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
        "method": "hybrid_rrf",
    }


def build_context(
    retrieval_result: dict[str, Any],
    max_chunks: int = 5,
) -> str:
    """
    Формує контекст для RAG.

    Кожен chunk отримує номер [1], [2], ...
    щоб LLM могла посилатися на джерело.
    """

    chunks = retrieval_result.get(
        "results",
        [],
    )

    parts = []

    for index, item in enumerate(
        chunks[:max_chunks],
        start=1,
    ):
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

        url = item.get(
            "url"
        )

        text = item.get(
            "text",
            "",
        ).strip()

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
            f"[{index}] {header}\n{text}"
        )

    return "\n\n".join(parts)


def format_sources(
    retrieval_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Формує список джерел для API/UI.
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
                "url": item.get(
                    "url"
                ),
                "section": item.get(
                    "section"
                ),
                "page": item.get(
                    "page"
                ),
                "score": item.get(
                    "rrf_score",
                    0.0,
                ),
            }
        )

    return sources
