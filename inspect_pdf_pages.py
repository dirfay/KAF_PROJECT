from __future__ import annotations

import json
from collections import defaultdict


KB_PATH = "data/knowledge_documents.jsonl"

PDF_SOURCES = [
    "master_standard_2021",
    "master_opp_2024",
    "bachelor_opp_2025",
    "bachelor_opp_2024",
    "presentation_pdf",
]


def main() -> None:
    grouped = defaultdict(list)

    with open(
        KB_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            grouped[
                record["source_id"]
            ].append(record)

    print("=" * 80)
    print("ПЕРЕВІРКА PDF PAGE METADATA")
    print("=" * 80)

    for source_id in PDF_SOURCES:
        rows = grouped.get(
            source_id,
            [],
        )

        pages = sorted(
            {
                record.get("page")
                for record in rows
                if record.get("page")
                is not None
            }
        )

        print()
        print(source_id)
        print(
            "  chunks:",
            len(rows),
        )
        print(
            "  pages:",
            pages,
        )
        print(
            "  unique pages:",
            len(pages),
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
