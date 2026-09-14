from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pymupdf
from bs4 import BeautifulSoup, Tag


BASE_DIR = Path(__file__).resolve().parent

SOURCES_PATH = BASE_DIR / "data" / "sources.json"
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
OUTPUT_PATH = BASE_DIR / "data" / "knowledge_documents.jsonl"

MIN_CHUNK_CHARS = 350
MAX_CHUNK_CHARS = 1800
TARGET_CHUNK_CHARS = 1200


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_inline_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_heading(text: str) -> str:
    text = normalize_inline_text(text)

    if len(text) > 300:
        return ""

    return text


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def remove_unwanted_html(root: Tag) -> None:
    unwanted_tags = [
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "iframe",
        "form",
        "nav",
        "footer",
        "header",
        "aside",
        "template",
    ]

    for tag_name in unwanted_tags:
        for tag in root.find_all(tag_name):
            tag.decompose()

    selectors = [
        ".cookie",
        ".cookies",
        ".cookie-banner",
        ".cookie-consent",
        ".menu",
        ".navbar",
        ".navigation",
        ".nav",
        ".sidebar",
        ".widget",
        ".footer",
        ".header",
        ".breadcrumb",
        ".breadcrumbs",
        ".social-share",
        ".share-buttons",
        ".comments",
        ".comment-respond",
        ".related-posts",
        ".related",
        ".advertisement",
        ".ads",
        ".advert",
    ]

    for selector in selectors:
        try:
            for tag in root.select(selector):
                tag.decompose()
        except Exception:
            pass


def find_main_container(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    selectors = [
        ".entry-content",
        ".site-content",
        ".content-area",
        ".main-content",
        ".page-content",
        ".post-content",
        "article",
        "main",
        "#content",
        ".elementor-widget-theme-post-content",
        ".wp-block-post-content",
    ]

    candidates: list[tuple[int, int, Tag]] = []

    for selector in selectors:
        for tag in soup.select(selector):
            text = normalize_whitespace(
                tag.get_text("\n", strip=True)
            )

            words = count_words(text)

            if words < 20:
                continue

            selector_bonus = {
                ".entry-content": 1000,
                ".elementor-widget-theme-post-content": 980,
                ".wp-block-post-content": 960,
                "article": 900,
                "main": 850,
                ".content-area": 800,
                "#content": 760,
                ".site-content": 700,
                ".main-content": 680,
                ".page-content": 660,
                ".post-content": 640,
            }.get(selector, 500)

            score = words + selector_bonus

            candidates.append(
                (
                    score,
                    words,
                    tag,
                )
            )

    if candidates:
        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best = candidates[0][2]

        best_text = normalize_whitespace(
            best.get_text("\n", strip=True)
        )

        if count_words(best_text) >= 40:
            return best

    fallback_candidates: list[tuple[int, Tag]] = []

    for tag in soup.find_all(
        ["section", "div", "article", "main"]
    ):
        text = normalize_whitespace(
            tag.get_text("\n", strip=True)
        )

        words = count_words(text)

        if words < 30:
            continue

        fallback_candidates.append(
            (
                words,
                tag,
            )
        )

    fallback_candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    if fallback_candidates:
        return fallback_candidates[0][1]

    if soup.body is not None:
        return soup.body

    return soup


def html_to_blocks(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")

    remove_unwanted_html(soup)

    main = find_main_container(soup)

    blocks: list[dict[str, str]] = []

    seen_texts: set[str] = set()

    elements = main.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "li",
            "blockquote",
            "td",
            "th",
        ]
    )

    for element in elements:
        text = normalize_inline_text(
            element.get_text(" ", strip=True)
        )

        if not text:
            continue

        if len(text) < 2:
            continue

        normalized_key = text.lower()

        if normalized_key in seen_texts:
            continue

        seen_texts.add(normalized_key)

        tag_name = element.name.lower()

        if tag_name in {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            heading = clean_heading(text)

            if heading:
                blocks.append(
                    {
                        "type": "heading",
                        "text": heading,
                    }
                )

        else:
            blocks.append(
                {
                    "type": "text",
                    "text": text,
                }
            )

    total_chars = sum(
        len(block["text"])
        for block in blocks
    )

    fallback_text = normalize_whitespace(
        main.get_text("\n", strip=True)
    )

    fallback_lines: list[str] = []
    seen_fallback: set[str] = set()

    for line in fallback_text.splitlines():
        line = normalize_inline_text(line)

        if not line:
            continue

        key = line.lower()

        if key in seen_fallback:
            continue

        seen_fallback.add(key)
        fallback_lines.append(line)

    fallback_text = "\n".join(
        fallback_lines
    )

    if (
        total_chars < 700
        and len(fallback_text) > total_chars
    ):
        return [
            {
                "type": "text",
                "text": fallback_text,
            }
        ]

    if not blocks and fallback_text:
        return [
            {
                "type": "text",
                "text": fallback_text,
            }
        ]

    return blocks


def pdf_to_blocks(path: Path) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []

    document = pymupdf.open(path)

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            text = normalize_whitespace(
                page.get_text("text")
            )

            if not text:
                continue

            blocks.append(
                {
                    "type": "page",
                    "text": text,
                    "page": str(page_number),
                }
            )
    finally:
        document.close()

    return blocks


def split_large_text(
    text: str,
    max_chars: int,
) -> list[str]:
    text = normalize_whitespace(text)

    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(
        r"\n{2,}",
        text,
    )

    pieces: list[str] = []
    current = ""

    for paragraph in paragraphs:
        paragraph = normalize_whitespace(paragraph)

        if not paragraph:
            continue

        if len(paragraph) > max_chars:
            sentences = re.split(
                r"(?<=[.!?])\s+",
                paragraph,
            )

            for sentence in sentences:
                sentence = sentence.strip()

                if not sentence:
                    continue

                if (
                    len(current)
                    + len(sentence)
                    + 1
                    <= max_chars
                ):
                    current = (
                        f"{current} {sentence}".strip()
                        if current
                        else sentence
                    )
                else:
                    if current:
                        pieces.append(current)

                    current = sentence

            continue

        if (
            len(current)
            + len(paragraph)
            + 2
            <= max_chars
        ):
            current = (
                f"{current}\n\n{paragraph}".strip()
                if current
                else paragraph
            )
        else:
            if current:
                pieces.append(current)

            current = paragraph

    if current:
        pieces.append(current)

    return pieces


def make_chunks(
    blocks: list[dict[str, str]],
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    current_parts: list[str] = []
    current_heading = ""
    current_page = ""

    def current_size() -> int:
        return len(
            "\n\n".join(current_parts)
        )

    def flush(force: bool = False) -> None:
        nonlocal current_parts

        if not current_parts:
            return

        text = "\n\n".join(
            current_parts
        ).strip()

        minimum = 120 if force else MIN_CHUNK_CHARS

        if len(text) >= minimum:
            chunks.append(
                {
                    "text": text,
                    "section": current_heading,
                    "page": current_page,
                }
            )

        current_parts = []

    for block in blocks:
        block_type = block.get(
            "type",
            "text",
        )

        text = normalize_inline_text(
            block.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        if block_type == "heading":
            if current_parts and current_size() >= TARGET_CHUNK_CHARS:
                flush()

            current_heading = clean_heading(
                text
            )

            continue

        page = block.get(
            "page",
            "",
        )

        if page:
            if (
                current_page
                and page != current_page
                and current_parts
            ):
                flush(force=True)

            current_page = page

        if len(text) > MAX_CHUNK_CHARS:
            subparts = split_large_text(
                text,
                MAX_CHUNK_CHARS,
            )

            for subpart in subparts:
                subpart = normalize_inline_text(
                    subpart
                )

                if not subpart:
                    continue

                if current_parts:
                    size = current_size()

                    if (
                        size
                        + len(subpart)
                        + 2
                        > TARGET_CHUNK_CHARS
                    ):
                        flush(force=True)

                current_parts.append(
                    subpart
                )

            continue

        if current_parts:
            size = current_size()

            if (
                size
                + len(text)
                + 2
                > TARGET_CHUNK_CHARS
            ):
                flush(force=True)

        current_parts.append(text)

    flush(force=True)

    return chunks


def normalize_for_hash(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def content_hash(text: str) -> str:
    normalized = normalize_for_hash(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def infer_metadata(
    source: dict[str, Any],
) -> dict[str, Any]:
    source_id = source["id"]
    title = source.get(
        "title",
        "",
    )

    metadata: dict[str, Any] = {
        "priority": source.get(
            "priority",
            "medium",
        ),
        "source_type": source.get(
            "type",
            "",
        ),
        "enabled": bool(
            source.get(
                "enabled",
                False,
            )
        ),
    }

    bachelor_ids = {
        "bachelor_programs",
        "f4_program",
        "data_analysis_124",
        "curriculum_bachelor",
        "disciplines_bachelor",
        "bachelor_opp_2025",
        "bachelor_opp_2024",
        "bachelor_standard_124",
    }

    master_ids = {
        "master_programs",
        "curriculum_master",
        "master_standard_2021",
        "master_opp_2024",
    }

    postgraduate_ids = {
        "academic_research_program",
        "curriculum_postgraduate",
        "postgraduate_department",
    }

    admission_ids = {
        "applicant_site",
        "admission_rules",
        "admission_documents",
        "tuition_prices",
        "preparatory_department",
        "nmt_calculator",
    }

    if source_id in bachelor_ids:
        metadata["education_level"] = "bachelor"

    elif source_id in master_ids:
        metadata["education_level"] = "master"

    elif source_id in postgraduate_ids:
        metadata["education_level"] = "postgraduate"

    elif source_id in admission_ids:
        metadata["category"] = "admission"

    else:
        metadata["category"] = "department"

    year_match = re.search(
        r"20(?:2[0-9])",
        f"{source_id} {title}",
    )

    if year_match:
        metadata["year"] = int(
            year_match.group(0)
        )

    if source_id == "bachelor_opp_2025":
        metadata["document_status"] = (
            "current_candidate"
        )

    elif source_id == "bachelor_opp_2024":
        metadata["document_status"] = (
            "historical"
        )

    elif source_id == "master_opp_2024":
        metadata["document_status"] = (
            "historical_or_reference"
        )

    elif source_id == "master_standard_2021":
        metadata["document_status"] = (
            "normative_standard"
        )

    elif source_id == "bachelor_standard_124":
        metadata["document_status"] = (
            "normative_standard"
        )

    else:
        metadata["document_status"] = "page"

    return metadata


def build_document_record(
    source: dict[str, Any],
    chunk: dict[str, Any],
    chunk_number: int,
) -> dict[str, Any]:
    text = normalize_whitespace(
        chunk["text"]
    )

    source_id = source["id"]

    record_id = (
        f"{source_id}_"
        f"{chunk_number:04d}_"
        f"{content_hash(text)[:12]}"
    )

    metadata = infer_metadata(
        source
    )

    record: dict[str, Any] = {
        "id": record_id,
        "source_id": source_id,
        "title": source.get(
            "title",
            source_id,
        ),
        "url": source.get(
            "url",
            "",
        ),
        "text": text,
        "metadata": metadata,
        "chunk_index": chunk_number,
    }

    section = normalize_inline_text(
        chunk.get(
            "section",
            "",
        )
    )

    if section:
        record["section"] = section

    page = chunk.get(
        "page",
        "",
    )

    if page:
        try:
            record["page"] = int(page)
        except ValueError:
            record["page"] = page

    return record


def main() -> None:
    if not SOURCES_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено: {SOURCES_PATH}"
        )

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Не знайдено: {DOCUMENTS_DIR}"
        )

    with SOURCES_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        sources = json.load(f)

    active_sources = [
        source
        for source in sources
        if source.get(
            "enabled",
            False,
        )
    ]

    source_map = {
        source["id"]: source
        for source in active_sources
    }

    print(
        "KAF_PROJECT — побудова "
        "knowledge_documents.jsonl"
    )

    print()
    print(
        f"Активних джерел: "
        f"{len(active_sources)}"
    )

    print(
        f"Папка документів: "
        f"{DOCUMENTS_DIR}"
    )

    print(
        f"Вихідний файл: "
        f"{OUTPUT_PATH}"
    )

    print()

    all_records: list[
        dict[str, Any]
    ] = []

    total_sources_ok = 0
    total_sources_missing = 0
    total_chunks = 0

    for source_id, source in source_map.items():
        source_type = source.get(
            "type",
            "",
        ).lower()

        if source_type == "pdf":
            file_path = (
                DOCUMENTS_DIR
                / f"{source_id}.pdf"
            )
        else:
            file_path = (
                DOCUMENTS_DIR
                / f"{source_id}.html"
            )

        print("=" * 80)
        print(
            f"SOURCE: {source_id}"
        )
        print(
            f"TITLE:  "
            f"{source.get('title', '')}"
        )
        print(
            f"FILE:   {file_path.name}"
        )

        if not file_path.exists():
            print(
                "STATUS: НЕ ЗНАЙДЕНО"
            )
            print()

            total_sources_missing += 1
            continue

        try:
            if source_type == "pdf":
                blocks = pdf_to_blocks(
                    file_path
                )
            else:
                html = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                blocks = html_to_blocks(
                    html
                )

            chunks = make_chunks(
                blocks
            )

            source_records: list[
                dict[str, Any]
            ] = []

            for index, chunk in enumerate(
                chunks
            ):
                record = build_document_record(
                    source,
                    chunk,
                    index,
                )

                source_records.append(
                    record
                )

            all_records.extend(
                source_records
            )

            source_chars = sum(
                len(record["text"])
                for record in source_records
            )

            total_chunks += len(
                source_records
            )

            total_sources_ok += 1

            print(
                f"BLOCKS:  "
                f"{len(blocks)}"
            )
            print(
                f"CHUNKS:  "
                f"{len(source_records)}"
            )
            print(
                f"CHARS:   "
                f"{source_chars}"
            )
            print(
                "STATUS:  OK"
            )

        except Exception as exc:
            print(
                "STATUS: ПОМИЛКА: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        print()

    unique_records: list[
        dict[str, Any]
    ] = []

    seen_keys: set[
        tuple[str, str]
    ] = set()

    for record in all_records:
        source_id = record[
            "source_id"
        ]

        text_hash = content_hash(
            record["text"]
        )

        dedup_key = (
            source_id,
            text_hash,
        )

        if dedup_key in seen_keys:
            continue

        seen_keys.add(
            dedup_key
        )

        unique_records.append(
            record
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for record in unique_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    final_chars = sum(
        len(record["text"])
        for record in unique_records
    )

    print("=" * 80)
    print("ЗАВЕРШЕНО")
    print("=" * 80)

    print(
        f"Джерел OK:              "
        f"{total_sources_ok}"
    )

    print(
        f"Джерел відсутніх:       "
        f"{total_sources_missing}"
    )

    print(
        f"Chunk до дедуплікації:  "
        f"{total_chunks}"
    )

    print(
        f"Chunk після дедуплікації: "
        f"{len(unique_records)}"
    )

    print(
        f"Символів у базі:        "
        f"{final_chars}"
    )

    print(
        f"Файл:                   "
        f"{OUTPUT_PATH}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
