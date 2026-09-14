from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SOURCES_PATH = BASE_DIR / "data" / "sources.json"
KB_PATH = BASE_DIR / "data" / "knowledge_documents.jsonl"


IMPORTANT_SOURCES = [
    "home",
    "staff",
    "news",
    "master_programs",
    "bachelor_programs",
    "f4_program",
    "data_analysis_124",
    "curriculum_bachelor",
    "curriculum_master",
    "survey_reports",
]

PDF_SOURCES = [
    "master_standard_2021",
    "master_opp_2024",
    "bachelor_opp_2025",
    "bachelor_opp_2024",
    "presentation_pdf",
]


def load_sources() -> list[dict]:
    with SOURCES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_records() -> list[dict]:
    records = []

    with KB_PATH.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"ПОМИЛКА JSON, рядок {line_number}: {exc}"
                )

    return records


def main() -> None:
    if not SOURCES_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено: {SOURCES_PATH}"
        )

    if not KB_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено: {KB_PATH}"
        )

    sources = load_sources()
    records = load_records()

    active_sources = [
        source
        for source in sources
        if source.get("enabled", False)
    ]

    active_ids = {
        source["id"]
        for source in active_sources
    }

    grouped = defaultdict(list)

    for record in records:
        grouped[record.get("source_id", "")].append(record)

    print("=" * 80)
    print("KAF_PROJECT — ПЕРЕВІРКА KNOWLEDGE BASE")
    print("=" * 80)

    print()
    print("ЗАГАЛЬНА ІНФОРМАЦІЯ")
    print("-" * 80)
    print("Усього джерел:", len(sources))
    print("Активних джерел:", len(active_sources))
    print("Записів/chunks:", len(records))
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

    bad_records = []

    for index, record in enumerate(records, start=1):
        missing = required_fields - set(record.keys())

        if missing:
            bad_records.append(
                {
                    "line": index,
                    "missing": sorted(missing),
                }
            )

    print("ФОРМАТ ЗАПИСІВ")
    print("-" * 80)
    print(
        "Записів із відсутніми обов'язковими полями:",
        len(bad_records),
    )

    if bad_records:
        for bad in bad_records[:10]:
            print(
                f"  Рядок {bad['line']}: "
                f"{bad['missing']}"
            )

    print()

    print("ПЕРЕВІРКА ДЖЕРЕЛ")
    print("-" * 80)

    missing_source_chunks = []

    for source_id in sorted(active_ids):
        count = len(grouped.get(source_id, []))

        if count == 0:
            missing_source_chunks.append(source_id)

    if missing_source_chunks:
        print("Джерела БЕЗ chunks:")

        for source_id in missing_source_chunks:
            print(f"  - {source_id}")
    else:
        print(
            "Усі активні джерела мають хоча б один chunk."
        )

    print()

    print("CHUNK ПО ДЖЕРЕЛАХ")
    print("-" * 80)

    for source_id in sorted(grouped):
        rows = grouped[source_id]

        chars = sum(
            len(record.get("text", ""))
            for record in rows
        )

        words = sum(
            len(record.get("text", "").split())
            for record in rows
        )

        print(
            f"{source_id:35} "
            f"chunks={len(rows):4} "
            f"chars={chars:8} "
            f"words={words:7}"
        )

    print()

    print("ВАЖЛИВІ СТОРІНКИ")
    print("-" * 80)

    for source_id in IMPORTANT_SOURCES:
        rows = grouped.get(source_id, [])

        chars = sum(
            len(record.get("text", ""))
            for record in rows
        )

        words = sum(
            len(record.get("text", "").split())
            for record in rows
        )

        print(
            f"{source_id:28} "
            f"chunks={len(rows):4} "
            f"chars={chars:8} "
            f"words={words:7}"
        )

    print()

    print("PDF")
    print("-" * 80)

    for source_id in PDF_SOURCES:
        rows = grouped.get(source_id, [])

        chars = sum(
            len(record.get("text", ""))
            for record in rows
        )

        pages = {
            record.get("page")
            for record in rows
            if record.get("page") is not None
        }

        print(
            f"{source_id:25} "
            f"chunks={len(rows):4} "
            f"chars={chars:8} "
            f"pages={len(pages):4}"
        )

    print()

    empty_text = [
        record
        for record in records
        if not record.get("text", "").strip()
    ]

    print("ПОРОЖНІ ТЕКСТИ")
    print("-" * 80)
    print("Порожніх записів:", len(empty_text))

    print()

    metadata_types = Counter(
        type(record.get("metadata")).__name__
        for record in records
    )

    print("METADATA")
    print("-" * 80)

    for name, count in sorted(metadata_types.items()):
        print(f"{name:15} {count}")

    print()

    print("РОЗПОДІЛ EDUCATION LEVEL")
    print("-" * 80)

    education_levels = Counter()

    for record in records:
        metadata = record.get("metadata", {})

        if isinstance(metadata, dict):
            level = metadata.get(
                "education_level",
                "(не вказано)",
            )

            education_levels[level] += 1

    for level, count in sorted(
        education_levels.items()
    ):
        print(f"{level:20} {count}")

    print()

    print("РОЗПОДІЛ DOCUMENT STATUS")
    print("-" * 80)

    document_status = Counter()

    for record in records:
        metadata = record.get("metadata", {})

        if isinstance(metadata, dict):
            status = metadata.get(
                "document_status",
                "(не вказано)",
            )

            document_status[status] += 1

    for status, count in sorted(
        document_status.items()
    ):
        print(f"{status:30} {count}")

    print()

    print("ПРИКЛАДИ CHUNK")
    print("-" * 80)

    for record in records[:5]:
        print()
        print("ID:", record.get("id"))
        print("SOURCE:", record.get("source_id"))
        print("TITLE:", record.get("title"))
        print("SECTION:", record.get("section", ""))
        print("PAGE:", record.get("page", ""))
        print("METADATA:", record.get("metadata"))
        print(
            "TEXT:",
            record.get("text", "")[:500]
            .replace("\n", " ")
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
