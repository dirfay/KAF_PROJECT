from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"


TARGETS = [
    "consultations.html",
    "academic_calendar.html",
    "master_programs.html",
    "bachelor_programs.html",
    "presentation.html",
    "survey_reports_master.html",
]


DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".csv",
}


def normalize_url(base_url: str, href: str) -> str:
    return urljoin(base_url, href.strip())


def inspect_file(file_path: Path) -> None:
    print()
    print("=" * 110)
    print(f"FILE: {file_path.name}")
    print("=" * 110)

    if not file_path.exists():
        print("Файл не знайдено.")
        return

    soup = BeautifulSoup(
        file_path.read_bytes(),
        "html.parser",
    )

    # Знаходимо всі посилання.
    links = []

    for link in soup.find_all("a"):
        href = link.get("href")

        if not href:
            continue

        href = href.strip()

        text = " ".join(
            link.get_text(" ", strip=True).split()
        )

        links.append(
            {
                "text": text,
                "href": href,
            }
        )

    print(f"Всього <a> посилань: {len(links)}")

    print()
    print("DOCUMENT LINKS")
    print("-" * 110)

    document_count = 0

    for item in links:
        href = item["href"]

        parsed = urlparse(href)

        path = parsed.path.lower()

        extension = Path(path).suffix

        if extension in DOCUMENT_EXTENSIONS:
            document_count += 1

            print(
                f"[{document_count}]"
            )

            print(
                f"TEXT: {item['text'] or '(без назви)'}"
            )

            print(
                f"HREF: {href}"
            )

            print()

    if document_count == 0:
        print("Документних посилань не знайдено.")

    print()
    print("SPECIAL / EMBEDDED ELEMENTS")
    print("-" * 110)

    iframe_count = 0

    for iframe in soup.find_all("iframe"):
        iframe_count += 1

        print(
            f"[IFRAME {iframe_count}] "
            f"src={iframe.get('src')}"
        )

        print(
            f"        title={iframe.get('title')}"
        )

    if iframe_count == 0:
        print("IFRAME не знайдено.")

    print()
    print("SCRIPT SOURCES / DATA")
    print("-" * 110)

    script_count = 0

    for script in soup.find_all("script"):
        src = script.get("src")

        if src:
            script_count += 1

            print(
                f"[SCRIPT {script_count}] "
                f"src={src}"
            )

    if script_count == 0:
        print("Зовнішніх script src не знайдено.")

    print()
    print("ALL EXTERNAL-LIKE HREFS")
    print("-" * 110)

    external_count = 0

    for item in links:
        href = item["href"]

        if (
            href.startswith("http://")
            or href.startswith("https://")
            or href.startswith("//")
        ):
            external_count += 1

            print(
                f"[{external_count}] "
                f"{item['text'] or '(без назви)'}"
            )

            print(
                f"    {href}"
            )

    if external_count == 0:
        print("Абсолютних зовнішніх href не знайдено.")


def main() -> None:
    print(
        "KAF_PROJECT — аналіз документів та embedded-ресурсів"
    )

    for filename in TARGETS:
        inspect_file(
            DOCUMENTS_DIR / filename
        )


if __name__ == "__main__":
    main()
