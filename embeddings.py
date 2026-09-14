from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_PATH = (
    BASE_DIR
    / "data"
    / "knowledge_documents.jsonl"
)

EMBEDDINGS_PATH = (
    BASE_DIR
    / "data"
    / "embeddings.json"
)

MODEL_NAME = "intfloat/multilingual-e5-small"

MODEL_VERSION = 1

BATCH_SIZE = 16

TOP_K = 10


_MODEL: SentenceTransformer | None = None


def load_knowledge() -> list[dict[str, Any]]:
    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(
            f"Не знайдено базу знань:\n{KNOWLEDGE_PATH}"
        )

    records: list[dict[str, Any]] = []

    with KNOWLEDGE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Помилка JSON у рядку "
                    f"{line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def build_passage_text(
    record: dict[str, Any],
) -> str:
    title = str(
        record.get(
            "title",
            "",
        )
    )

    section = str(
        record.get(
            "section",
            "",
        )
    )

    text = str(
        record.get(
            "text",
            "",
        )
    )

    parts = []

    if title:
        parts.append(title)

    if section:
        parts.append(section)

    if text:
        parts.append(text)

    return "\n".join(parts).strip()


def get_model(
    show_message: bool = True,
) -> SentenceTransformer:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    if show_message:
        print()
        print("=" * 80)
        print("SEMANTIC MODEL — LOAD")
        print("=" * 80)

        print()
        print("Модель:")
        print(MODEL_NAME)

        print()
        print(
            "Завантаження моделі..."
        )

    started = time.perf_counter()

    _MODEL = SentenceTransformer(
        MODEL_NAME
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if show_message:
        print(
            f"Модель завантажена "
            f"за {elapsed:.2f} с"
        )

    return _MODEL


def build_embeddings() -> None:
    records = load_knowledge()

    if not records:
        raise ValueError(
            "База знань порожня."
        )

    model = get_model()

    texts = [
        build_passage_text(record)
        for record in records
    ]

    print()
    print("=" * 80)
    print("SEMANTIC EMBEDDINGS — BUILD")
    print("=" * 80)

    print()
    print("Knowledge base:")
    print(KNOWLEDGE_PATH)

    print()
    print(
        f"Записів: {len(records)}"
    )

    print()
    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print()
    print(
        "Створення embeddings..."
    )

    started = time.perf_counter()

    embeddings = model.encode(
        [
            f"passage: {text}"
            for text in texts
        ],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    print()
    print(
        f"Час кодування: "
        f"{elapsed:.2f} с"
    )

    if len(embeddings) != len(records):
        raise RuntimeError(
            "Кількість embeddings "
            "не збігається "
            "з кількістю записів."
        )

    dimension = int(
        embeddings.shape[1]
    )

    output_records = []

    for record, vector in zip(
        records,
        embeddings,
    ):
        output_records.append(
            {
                "id": record.get("id"),
                "source_id": record.get(
                    "source_id"
                ),
                "title": record.get(
                    "title"
                ),
                "section": record.get(
                    "section"
                ),
                "page": record.get(
                    "page"
                ),
                "text": record.get(
                    "text"
                ),
                "metadata": record.get(
                    "metadata",
                    {},
                ),
                "embedding": [
                    float(value)
                    for value in vector
                ],
            }
        )

    output = {
        "version": MODEL_VERSION,
        "model": MODEL_NAME,
        "dimension": dimension,
        "records_count": len(
            output_records
        ),
        "normalized": True,
        "records": output_records,
    }

    EMBEDDINGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with EMBEDDINGS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
        )

    file_size_mb = (
        EMBEDDINGS_PATH.stat().st_size
        / (1024 * 1024)
    )

    print()
    print(
        f"Розмір embedding: "
        f"{dimension}"
    )

    print(
        f"Створено embeddings: "
        f"{len(output_records)}"
    )

    print(
        f"Файл: "
        f"{EMBEDDINGS_PATH}"
    )

    print(
        f"Розмір файлу: "
        f"{file_size_mb:.2f} MB"
    )

    print()
    print("Статус: OK")
    print("=" * 80)


def load_embeddings() -> dict[str, Any]:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Файл embeddings ще не створений:\n"
            f"{EMBEDDINGS_PATH}\n\n"
            "Спочатку виконайте:\n"
            "python embeddings.py build"
        )

    with EMBEDDINGS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Вектори мають різну розмірність."
        )

    return sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b,
        )
    )


def semantic_search(
    query: str,
    top_k: int = TOP_K,
    model: SentenceTransformer | None = None,
) -> list[dict[str, Any]]:
    query = query.strip()

    if not query:
        return []

    data = load_embeddings()

    if model is None:
        model = get_model()

    query_text = (
        f"query: {query}"
    )

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    query_vector = [
        float(value)
        for value in query_embedding
    ]

    results: list[dict[str, Any]] = []

    for record in data["records"]:
        score = cosine_similarity(
            query_vector,
            record["embedding"],
        )

        results.append(
            {
                "score": float(score),
                "record": record,
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]


def print_search_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 80)
    print("SEMANTIC SEARCH — E2")
    print("=" * 80)

    print()
    print("Запит:")
    print(query)

    print()
    print(
        f"TOP-{len(results)}"
    )

    print()
    print("-" * 80)

    for position, item in enumerate(
        results,
        start=1,
    ):
        record = item["record"]

        print(
            f"#{position} "
            f"similarity="
            f"{item['score']:.6f}"
        )

        print(
            "ID:",
            record.get("id"),
        )

        print(
            "SOURCE:",
            record.get("source_id"),
        )

        print(
            "TITLE:",
            record.get("title"),
        )

        if record.get("section"):
            print(
                "SECTION:",
                record.get("section"),
            )

        if record.get("page"):
            print(
                "PAGE:",
                record.get("page"),
            )

        print()
        print(
            str(
                record.get(
                    "text",
                    "",
                )
            )[:1200]
        )

        print("-" * 80)


def search_command(
    query: str,
) -> None:
    results = semantic_search(
        query,
        top_k=TOP_K,
    )

    print_search_results(
        query,
        results,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Використання:"
        )
        print(
            "  python embeddings.py build"
        )
        print(
            '  python embeddings.py search "текст"'
        )
        return

    command = sys.argv[1].lower()

    if command == "build":
        build_embeddings()
        return

    if command == "search":
        if len(sys.argv) < 3:
            raise SystemExit(
                "Потрібно вказати пошуковий запит."
            )

        query = " ".join(
            sys.argv[2:]
        )

        search_command(query)
        return

    raise SystemExit(
        f"Невідома команда: {command}"
    )


if __name__ == "__main__":
    main()
