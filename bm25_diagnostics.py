from __future__ import annotations

from bm25_index import search


TEST_QUERIES = [
    "контакти кафедри",
    "номер телефону кафедри",
    "хто працює на кафедрі",
    "новини кафедри",
    "навчальні дисципліни бакалавра",
    "освітня програма системний аналіз 2024",
    "освітня програма інтелектуальний аналіз даних 2024",
    "які документи потрібні для вступу",
    "які предмети вивчаються",
    "матеріально технічна база кафедри",
]


def main() -> None:
    print("=" * 80)
    print("BM25 — ДІАГНОСТИКА")
    print("=" * 80)

    for query in TEST_QUERIES:
        results = search(
            query,
            top_k=5,
        )

        print()
        print("QUERY:", query)

        for result in results:
            print(
                f"  #{result['rank']} "
                f"{result['source_id']:30} "
                f"{result['score']:.4f}"
            )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
