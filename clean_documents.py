from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SOURCES_PATH = BASE_DIR / "data" / "sources.json"
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"


def main() -> None:
    if not SOURCES_PATH.exists():
        raise FileNotFoundError(f"Не знайдено: {SOURCES_PATH}")

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(f"Не знайдено: {DOCUMENTS_DIR}")

    with SOURCES_PATH.open("r", encoding="utf-8") as f:
        sources = json.load(f)

    active_sources = [
        source
        for source in sources
        if source.get("enabled", False)
    ]

    expected_names = set()

    for source in active_sources:
        source_id = source["id"]
        source_type = source["type"].lower()

        if source_type == "pdf":
            expected_names.add(f"{source_id}.pdf")
        else:
            expected_names.add(f"{source_id}.html")

    print("KAF_PROJECT — очищення локальних документів")
    print()
    print(f"Активних джерел: {len(active_sources)}")
    print(f"Очікуваних файлів: {len(expected_names)}")
    print(f"Папка: {DOCUMENTS_DIR}")
    print()

    removed = 0
    kept = 0

    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if not path.is_file():
            continue

        if path.name in expected_names:
            print(f"KEEP    {path.name}")
            kept += 1
        else:
            print(f"DELETE  {path.name}")
            path.unlink()
            removed += 1

    print()
    print("=" * 80)
    print(f"Залишено:  {kept}")
    print(f"Видалено:  {removed}")
    print(f"Очікується: {len(expected_names)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
