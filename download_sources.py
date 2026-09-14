from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


BASE_DIR = Path(__file__).resolve().parent
SOURCES_PATH = BASE_DIR / "data" / "sources.json"
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

REQUEST_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0 Safari/537.36 "
    "KAF_PROJECT-Knowledge-Ingest/1.0"
)


def load_sources() -> list[dict]:
    if not SOURCES_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено файл джерел: {SOURCES_PATH}"
        )

    with SOURCES_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("sources.json повинен містити JSON-масив.")

    result = []

    for source in data:
        if not isinstance(source, dict):
            continue

        if not source.get("enabled", False):
            continue

        if not source.get("url"):
            continue

        result.append(source)

    return result


def safe_filename(source: dict) -> str:
    source_id = str(source.get("id", "source")).strip()

    source_type = str(source.get("type", "web")).strip().lower()

    parsed = urlparse(source["url"])

    suffix = ".pdf" if source_type == "pdf" else ".html"

    if parsed.path.endswith(".pdf"):
        suffix = ".pdf"

    name = re.sub(r"[^a-zA-Z0-9а-яА-ЯіІїЇєЄґҐ_-]+", "_", source_id)

    return f"{name}{suffix}"


def get_content_type(response: requests.Response) -> str:
    return response.headers.get("Content-Type", "").lower()


def download_source(session: requests.Session, source: dict) -> tuple[bool, str]:
    url = source["url"]
    destination = DOCUMENTS_DIR / safe_filename(source)

    print()
    print("=" * 80)
    print(f"ID:      {source.get('id')}")
    print(f"TITLE:   {source.get('title')}")
    print(f"TYPE:    {source.get('type')}")
    print(f"URL:     {url}")
    print(f"TARGET:  {destination}")
    print("=" * 80)

    try:
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        print(f"HTTP:    {response.status_code}")
        print(f"FINAL:   {response.url}")
        print(f"TYPE:    {get_content_type(response)}")
        print(f"SIZE:    {len(response.content)} bytes")

        response.raise_for_status()

        if not response.content:
            return False, "Відповідь порожня."

        destination.write_bytes(response.content)

        print(f"OK:      збережено {destination}")

        return True, "OK"

    except requests.RequestException as exc:
        print(f"ERROR:   {exc}")
        return False, str(exc)

    except OSError as exc:
        print(f"ERROR:   неможливо записати файл: {exc}")
        return False, str(exc)


def main() -> int:
    print("KAF_PROJECT — завантаження джерел бази знань")
    print()

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        sources = load_sources()
    except Exception as exc:
        print(f"ПОМИЛКА: {exc}")
        return 1

    print(f"Активних джерел: {len(sources)}")
    print(f"Папка документів: {DOCUMENTS_DIR}")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    success_count = 0
    failed_count = 0

    results = []

    for index, source in enumerate(sources, start=1):
        print()
        print(f"[{index}/{len(sources)}] Завантаження...")

        success, message = download_source(session, source)

        results.append(
            {
                "id": source.get("id"),
                "title": source.get("title"),
                "url": source.get("url"),
                "type": source.get("type"),
                "success": success,
                "message": message,
            }
        )

        if success:
            success_count += 1
        else:
            failed_count += 1

        # Невелика пауза між запитами,
        # щоб не створювати непотрібне навантаження на сайт.
        time.sleep(1)

    report_path = BASE_DIR / "data" / "download_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print()
    print("=" * 80)
    print("ЗАВЕРШЕНО")
    print("=" * 80)
    print(f"Успішно:       {success_count}")
    print(f"Помилок:       {failed_count}")
    print(f"Всього:        {len(sources)}")
    print(f"Звіт:          {report_path}")
    print("=" * 80)

    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
