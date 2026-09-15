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
    print("RETRIEVAL PIPELINE TEST")
    print("=" * 80)

    print()
    print("QUERY:")
    print(result["query"])

    print()
    print(
        "LATENCY:",
        f"{result['latency_seconds']:.3f}s",
    )

    print()
    print("RESULTS:")

    for item in result["results"]:
        print(
            f"#{item['rank']} "
            f"{item['source_id']} "
            f"reranker="
            f"{item['reranker_score']:.6f}"
        )

    print()
    print("SOURCES:")

    for source in format_sources(result):
        print(
            source["source_id"],
            "—",
            source["title"],
        )

    print()
    print("CONTEXT:")
    print(
        build_context(
            result,
            max_chunks=3,
        )
    )


if __name__ == "__main__":
    main()
