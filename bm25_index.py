from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parent

KB_PATH = (
    BASE_DIR
    / "data"
    / "knowledge_documents.jsonl"
)

INDEX_PATH = (
    BASE_DIR
    / "data"
    / "bm25_index.json"
)


# Українські та загальні службові слова,
# які майже не допомагають визначити релевантний документ.
UKRAINIAN_STOPWORDS = {
    "і",
    "й",
    "та",
    "а",
    "але",
    "або",
    "чи",
    "бо",
    "що",
    "як",
    "які",
    "який",
    "яка",
    "яке",
    "які",
    "це",
    "цей",
    "ця",
    "ці",
    "той",
    "такий",
    "така",
    "такі",
    "до",
    "з",
    "зі",
    "із",
    "за",
    "на",
    "у",
    "в",
    "по",
    "для",
    "про",
    "від",
    "при",
    "між",
    "без",
    "під",
    "над",
    "після",
    "перед",
    "через",
    "уже",
    "ще",
    "не",
    "ні",
    "так",
    "дуже",
    "можна",
    "потрібно",
    "потрібні",
    "є",
    "бути",
    "є",
    "зміст",
    "вступу",
}


# Поля, які не повинні мати таку саму вагу,
# як власне зміст документа.
BOILERPLATE_MARKERS = {
    "приймальна комісія",
    "публічна інформація",
    "швидкі посилання",
    "член європейської асоціації університетів",
    "нормативні документи",
    "зворотний зв’язок",
    "зворотний зв'язок",
}


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    text = text.lower()

    text = text.replace("’", "'")
    text = text.replace("ʼ", "'")
    text = text.replace("`", "'")
    text = text.replace("´", "'")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(
    text: str,
    remove_stopwords: bool = True,
) -> list[str]:
    text = normalize_text(text)

    tokens = re.findall(
        r"[a-zа-яіїєґ0-9]+(?:'[a-zа-яіїєґ0-9]+)*",
        text,
        flags=re.IGNORECASE,
    )

    if remove_stopwords:
        tokens = [
            token
            for token in tokens
            if token not in UKRAINIAN_STOPWORDS
            and len(token) > 1
        ]

    return tokens


def contains_boilerplate(text: str) -> bool:
    normalized = normalize_text(text)

    matches = 0

    for marker in BOILERPLATE_MARKERS:
        if marker in normalized:
            matches += 1

    return matches >= 2


def load_records() -> list[dict[str, Any]]:
    if not KB_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено knowledge base: {KB_PATH}"
        )

    records = []

    with KB_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Помилка JSONL у рядку "
                    f"{line_number}: {exc}"
                ) from exc

            if not record.get("text"):
                continue

            records.append(record)

    return records


def searchable_text(
    record: dict[str, Any],
) -> str:
    title = record.get(
        "title",
        "",
    )

    section = record.get(
        "section",
        "",
    )

    text = record.get(
        "text",
        "",
    )

    # Назву документа і заголовок секції
    # повторюємо кілька разів, щоб BM25
    # надавав їм підвищену вагу.
    return " ".join(
        [
            title,
            title,
            section,
            section,
            text,
        ]
    )


def build_corpus(
    records: list[dict[str, Any]],
) -> list[list[str]]:
    corpus = []

    for record in records:
        text = searchable_text(record)

        tokens = tokenize(
            text,
            remove_stopwords=True,
        )

        corpus.append(tokens)

    return corpus


def build_bm25(
    records: list[dict[str, Any]],
) -> BM25Okapi:
    corpus = build_corpus(records)

    return BM25Okapi(corpus)


def build_index_data(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    corpus = build_corpus(records)

    return {
        "version": 2,
        "tokenizer": "ukrainian_stopwords",
        "title_weight": 2,
        "section_weight": 2,
        "documents": corpus,
        "record_ids": [
            record["id"]
            for record in records
        ],
    }


def save_index_data(
    index_data: dict[str, Any],
) -> None:
    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with INDEX_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            index_data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def build_command() -> None:
    records = load_records()

    index_data = build_index_data(
        records
    )

    save_index_data(
        index_data
    )

    print("=" * 80)
    print("BM25 INDEX — BUILD")
    print("=" * 80)
    print()
    print("Knowledge base:")
    print(KB_PATH)
    print()
    print("Записів:", len(records))
    print(
        "Документів у BM25:",
        len(index_data["documents"]),
    )
    print(
        "Версія індексу:",
        index_data["version"],
    )
    print(
        "Токенізація:",
        index_data["tokenizer"],
    )
    print()
    print("Статус: OK")
    print("=" * 80)


def result_from_record(
    record: dict[str, Any],
    rank: int,
    score: float,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": float(score),
        "id": record["id"],
        "source_id": record["source_id"],
        "title": record["title"],
        "url": record["url"],
        "section": record.get(
            "section",
            "",
        ),
        "page": record.get(
            "page",
            "",
        ),
        "text": record["text"],
    }


def search(
    query: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    records = load_records()

    if not records:
        return []

    bm25 = build_bm25(
        records
    )

    query_tokens = tokenize(
        query,
        remove_stopwords=True,
    )

    if not query_tokens:
        return []

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    results = []

    for index in ranked_indices:
        record = records[index]

        score = float(
            scores[index]
        )

        text = record.get(
            "text",
            "",
        )

        # Якщо chunk складається в основному
        # з глобального меню/службового шаблону,
        # трохи знижуємо його пріоритет.
        if contains_boilerplate(text):
            score *= 0.70

        # Додаткове підсилення прямого збігу
        # термінів у title.
        title_tokens = set(
            tokenize(
                record.get(
                    "title",
                    "",
                ),
                remove_stopwords=True,
            )
        )

        section_tokens = set(
            tokenize(
                record.get(
                    "section",
                    "",
                ),
                remove_stopwords=True,
            )
        )

        query_set = set(
            query_tokens
        )

        title_overlap = len(
            query_set & title_tokens
        )

        section_overlap = len(
            query_set & section_tokens
        )

        score += (
            1.5
            * title_overlap
        )

        score += (
            0.75
            * section_overlap
        )

        results.append(
            result_from_record(
                record,
                0,
                score,
            )
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # Проставляємо фінальний rank.
    for rank, result in enumerate(
        results[:top_k],
        start=1,
    ):
        result["rank"] = rank

    return results[:top_k]


def print_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 80)
    print("BM25 SEARCH — E1")
    print("=" * 80)
    print()
    print("Запит:")
    print(query)
    print()
    print("Токени:")
    print(
        tokenize(
            query,
            remove_stopwords=True,
        )
    )
    print()

    if not results:
        print("Результатів немає.")
        return

    for result in results:
        print("-" * 80)

        print(
            f"#{result['rank']} "
            f"score={result['score']:.4f}"
        )

        print(
            "ID:",
            result["id"],
        )

        print(
            "SOURCE:",
            result["source_id"],
        )

        print(
            "TITLE:",
            result["title"],
        )

        if result["section"]:
            print(
                "SECTION:",
                result["section"],
            )

        if result["page"] != "":
            print(
                "PAGE:",
                result["page"],
            )

        print()
        print(
            result["text"][:700]
            .replace("\n", " ")
        )

    print()
    print("=" * 80)


def interactive_command() -> None:
    print(
        "BM25 interactive search. "
        "Для виходу введіть: exit"
    )

    while True:
        try:
            query = input(
                "\nЗапит: "
            ).strip()
        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print()
            break

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = search(
            query,
            top_k=10,
        )

        print_results(
            query,
            results,
        )


def main() -> None:
    if len(sys.argv) < 2:
        print()
        print(
            "Використання:"
        )
        print(
            "  python bm25_index.py build"
        )
        print(
            '  python bm25_index.py search "текст запиту"'
        )
        print(
            "  python bm25_index.py interactive"
        )
        print()
        return

    command = sys.argv[1].lower()

    if command == "build":
        build_command()
        return

    if command == "search":
        if len(sys.argv) < 3:
            raise SystemExit(
                "Після search потрібен запит."
            )

        query = " ".join(
            sys.argv[2:]
        )

        results = search(
            query,
            top_k=10,
        )

        print_results(
            query,
            results,
        )

        return

    if command == "interactive":
        interactive_command()
        return

    raise SystemExit(
        f"Невідома команда: {command}"
    )


if __name__ == "__main__":
    main()
