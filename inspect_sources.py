from __future__ import annotations

import json
from collections import defaultdict


KB_PATH = "data/knowledge_documents.jsonl"

TARGETS = [
    "home",
    "staff",
    "news",
    "master_programs",
    "bachelor_programs",
    "f4_program",
    "data_analysis_124",
    "curriculum_bachelor",
    "curriculum_master",
    "academic_research_program",
    "infrastructure",
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
    print("ПЕРЕВІРКА ДЖЕРЕЛ")
    print("=" * 80)

    for source_id in TARGETS:
        rows = grouped.get(
            source_id,
            [],
        )

        chars = sum(
            len(record.get("text", ""))
            for record in rows
        )

        words = sum(
            len(
                record.get(
                    "text",
                    "",
                ).split()
            )
            for record in rows
        )

        print(
            f"{source_id:30} "
            f"chunks={len(rows):4} "
            f"chars={chars:8} "
            f"words={words:7}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
