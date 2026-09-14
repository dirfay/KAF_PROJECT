from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SOURCES_PATH = BASE_DIR / "data" / "sources.json"
KB_PATH = BASE_DIR / "data" / "knowledge_documents.jsonl"


def main() -> None:
    if not SOURCES_PATH.exists():
        raise FileNotFoundError(f"Не знайдено файл: {SOURCES_PATH}")

    if not KB_PATH.exists():
        raise FileNotFoundError(f"Не знайдено файл: {KB_PATH}")

    with SOURCES_PATH.open("r", encoding="utf-8") as f:
        sources = json.load(f)

    active_source_ids = {
        source["id"]
        for source in sources
        if source.get("enabled", False)
    }

    records = []

    with KB_PATH.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"ПОМИЛКА JSONL, рядок {line_number}: {exc}")

    by_source = Counter(
        record.get("source_id", "")
        for record in records
    )

    print("=" * 80)
    print("ПЕРЕВІРКА KNOWLEDGE BASE")
    print("=" * 80)

    print(f"Активних джерел: {len(active_source_ids)}")
    print(f"Записів/chunks:  {len(records)}")
    print()

    required_fields = {
        "id",
        "source_id",
        "title",
        "url",
        "text",
        "metadata",
        "chunk_index",
    }

    records_with_missing_fields = []

    for index, record in enumerate(records, start=1):
        missing = sorted(required_fields - set(record.keys()))

        if missing:
            records_with_missing_fields.append(
                (index, missing)
            )

    print(
        "Записів із відсутніми полями: "
        f"{len(records_with_missing_fields)}"
    )

    if records_with_missing_fields:
        for index, missing in records_with_missing_fields[:10]:
            print(f"  Рядок {index}: {missing}")

        if len(records_with_missing_fields) > 10:
            print(
                f"  ... ще "
                f"{len(records_with_missing_fields) - 10}"
            )

    print()

    missing_sources = sorted(
        source_id
        for source_id in active_source_ids
        if by_source[source_id] == 0
    )

    if missing_sources:
        print("АКТИВНІ ДЖЕРЕЛА БЕЗ CHUNK:")

        for source_id in missing_sources:
            print(f"  - {source_id}")
    else:
        print("Усі активні джерела мають хоча б один chunk.")

    print()

    print("КІЛЬКІСТЬ CHUNK ПО ДЖЕРЕЛАХ:")

    for source_id in sorted(by_source):
        print(f"  {source_id:35} {by_source[source_id]}")

    print()

    empty_text = [
        record
        for record in records
        if not str(record.get("text", "")).strip()
    ]

    print(f"Порожніх записів: {len(empty_text)}")

    print()

    metadata_counter = Counter(
        type(record.get("metadata")).__name__
        for record in records
    )

    print("Тип metadata:")

    for metadata_type, count in sorted(metadata_counter.items()):
        print(f"  {metadata_type}: {count}")

    print()

    if records:
        print("ПЕРШИЙ ЗАПИС:")
        print("-" * 80)

        first = records[0]

        print("ID:", first.get("id"))
        print("SOURCE:", first.get("source_id"))
        print("TITLE:", first.get("title"))
        print("URL:", first.get("url"))
        print("SECTION:", first.get("section", ""))
        print("PAGE:", first.get("page", ""))
        print("CHUNK INDEX:", first.get("chunk_index"))
        print("METADATA:", first.get("metadata"))

        print("TEXT:")
        print(first.get("text", "")[:1000])

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
