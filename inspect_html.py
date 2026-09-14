from __future__ import annotations

from pathlib import Path
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"


TARGETS = [
    "home.html",
    "contacts.html",
    "bachelor_programs.html",
    "master_programs.html",
    "staff.html",
    "news.html",
    "consultations.html",
    "presentation.html",
    "academic_calendar.html",
    "survey_reports_master.html",
]


def clean_text(text: str) -> str:
    return " ".join(text.split())


def inspect_file(path: Path) -> None:
    print()
    print("=" * 100)
    print(f"FILE: {path.name}")
    print("=" * 100)

    if not path.exists():
        print("Файл не знайдено.")
        return

    soup = BeautifulSoup(
        path.read_bytes(),
        "html.parser",
    )

    print(f"HTML size: {path.stat().st_size:,} bytes")

    if soup.title:
        print(
            f"TITLE: {clean_text(soup.title.get_text(' ', strip=True))}"
        )

    selectors = [
        "main",
        "article",
        ".entry-content",
        ".post-content",
        ".elementor",
        ".elementor-widget-theme-post-content",
        ".elementor-widget-container",
        ".site-main",
        "#content",
        ".content",
        ".page-content",
        "body",
    ]

    print()
    print("SELECTOR ANALYSIS")
    print("-" * 100)

    seen = set()

    for selector in selectors:
        elements = soup.select(selector)

        if not elements:
            continue

        for index, element in enumerate(elements[:10], start=1):
            text = clean_text(
                element.get_text(" ", strip=True)
            )

            marker = (
                selector,
                index,
                len(text),
            )

            if marker in seen:
                continue

            seen.add(marker)

            print(
                f"{selector:<45} "
                f"#{index:<3} "
                f"chars={len(text):>7,} "
                f"words={len(text.split()):>6,}"
            )

    print()
    print("LARGEST DIVS")
    print("-" * 100)

    divs = soup.find_all("div")

    candidates = []

    for div in divs:
        text = clean_text(
            div.get_text(" ", strip=True)
        )

        if not text:
            continue

        classes = " ".join(div.get("class", []))

        candidates.append(
            {
                "tag": "div",
                "classes": classes,
                "chars": len(text),
                "words": len(text.split()),
                "text": text[:250],
            }
        )

    candidates.sort(
        key=lambda item: item["chars"],
        reverse=True,
    )

    for item in candidates[:20]:
        print(
            f"chars={item['chars']:>7,} "
            f"words={item['words']:>6,} "
            f"class={item['classes'][:60]!r}"
        )
        print(
            f"    {item['text']}"
        )


def main() -> None:
    print(
        "KAF_PROJECT — діагностика структури HTML"
    )

    for filename in TARGETS:
        inspect_file(
            DOCUMENTS_DIR / filename
        )


if __name__ == "__main__":
    main()
