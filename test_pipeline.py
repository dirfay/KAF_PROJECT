from retrieval_pipeline import (
    build_context,
    format_sources,
    retrieve,
)


def main() -> None:
    query = "хто викладає на кафедрі"

    result = retrieve(
        query,
        candidates=20,
        top_k=5,
    )

    print("=" * 80)
    print("REAL RETRIEVAL PIPELINE TEST — E3")
    print("=" * 80)

    print()
    print("QUERY:")
    print(result["query"])

    print()
    print("METHOD:")
    print(result["method"])

    print()
    print("LATENCY:")
    print(
        f"{result['latency_seconds']:.3f}s"
    )

    print()
    print("RESULTS:")

    for item in result["results"]:
        print(
            f"#{item['rank']} "
            f"{item['source_id']} | "
            f"RRF={item['rrf_score']:.6f} | "
            f"BM25={item['bm25_rank']} | "
            f"Semantic={item['semantic_rank']}"
        )

    print()
    print("SOURCES:")

    for source in format_sources(
        result
    ):
        print(
            f"- {source['source_id']} — "
            f"{source['title']}"
        )

        if source.get("url"):
            print(
                f"  URL: {source['url']}"
            )

        if source.get("page"):
            print(
                f"  Page: {source['page']}"
            )

    print()
    print("CONTEXT:")
    print("-" * 80)

    print(
        build_context(
            result,
            max_chunks=5,
        )
    )


if __name__ == "__main__":
    main()
