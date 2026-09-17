from __future__ import annotations

import re
import time
from typing import Any

import requests

from retrieval_pipeline import (
    build_context,
    format_sources,
    retrieve,
)


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b-instruct"

REQUEST_TIMEOUT = 300

RAG_CONTEXT_CHUNKS = 2

MAX_CONTEXT_BLOCK_CHARS = 1000

NUM_PREDICT = 160

NUM_CTX = 2048


SYSTEM_PROMPT = (
    "Ти інформаційний асистент кафедри систем "
    "штучного інтелекту та аналізу даних ТНТУ. "
    "Відповідай лише за КОНТЕКСТОМ. "
    "Не вигадуй фактів. "
    "Якщо даних недостатньо, скажи про це. "
    "Відповідай українською. "
    "Не показуй міркування. "
    "Джерела позначай тільки номерами [1], [2]."
)


def _is_staff_question(question: str) -> bool:
    """
    Визначає запити, для яких основним джерелом
    має бути сторінка колективу кафедри.
    """

    question = question.lower()

    patterns = (
        "хто викладає",
        "хто викладач",
        "хто працює на кафедрі",
        "викладачі кафедри",
        "викладачі",
        "викладач",
        "колектив кафедри",
        "працівники кафедри",
        "співробітники кафедри",
        "склад кафедри",
        "хто входить до складу кафедри",
    )

    return any(pattern in question for pattern in patterns)


def _select_rag_results(
    question: str,
    retrieval: dict[str, Any],
    max_chunks: int = RAG_CONTEXT_CHUNKS,
) -> dict[str, Any]:
    """
    Вибирає найкорисніші результати для RAG-контексту.

    Сам hybrid retrieval не змінюється.
    Ми лише формуємо компактний контекст для LLM.
    """

    results = list(
        retrieval.get(
            "results",
            [],
        )
    )

    if not results:
        selected = dict(retrieval)
        selected["results"] = []
        return selected

    selected_results: list[dict[str, Any]] = []

    if _is_staff_question(question):
        staff_results = [
            result
            for result in results
            if result.get("source_id") == "staff"
        ]

        other_results = [
            result
            for result in results
            if result.get("source_id") != "staff"
        ]

        selected_results = (
            staff_results + other_results
        )[:max_chunks]

    else:
        selected_results = results[:max_chunks]

    selected = dict(retrieval)
    selected["results"] = selected_results

    return selected


def _remove_urls_from_context(
    context: str,
) -> str:
    """
    Прибирає URL із контексту перед передачею LLM.

    URL залишаються в sources.
    """

    if not context:
        return ""

    context = re.sub(
        r"\s*\|\s*url=https?://\S+",
        "",
        context,
        flags=re.IGNORECASE,
    )

    context = re.sub(
        r"^\s*https?://\S+\s*$",
        "",
        context,
        flags=re.MULTILINE,
    )

    return context.strip()


def _compact_context(
    context: str,
    max_block_chars: int = MAX_CONTEXT_BLOCK_CHARS,
) -> str:
    """
    Додатково стискає контекст перед LLM.

    Контекст build_context складається з блоків:
    [1] ...
    [2] ...

    Кожен блок обмежується за кількістю символів.
    """

    context = (context or "").strip()

    if not context:
        return ""

    context = _remove_urls_from_context(context)

    blocks = re.split(
        r"\n\s*\n(?=\[\d+\]\s)",
        context,
    )

    compact_blocks: list[str] = []

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        if len(block) > max_block_chars:
            block = (
                block[:max_block_chars]
                .rsplit(" ", 1)[0]
                .strip()
                + "..."
            )

        compact_blocks.append(block)

    return "\n\n".join(compact_blocks)


def _build_prompt(
    question: str,
    context: str,
) -> str:
    return (
        "КОНТЕКСТ:\n"
        f"{context}\n\n"
        "ПИТАННЯ:\n"
        f"{question}\n\n"
        "Дай коротку точну відповідь українською. "
        "Використовуй тільки КОНТЕКСТ. "
        "Не додавай припущень. "
        "Якщо наводиш факти з джерел, "
        "постав відповідний номер [1] або [2]."
    )


def _clean_answer(
    answer: str,
) -> str:
    """
    Видаляє можливі службові фрагменти моделі.
    """

    answer = (answer or "").strip()

    if "<think>" in answer:
        answer = answer.split(
            "</think>",
            1,
        )[-1].strip()

    if "</think>" in answer:
        answer = answer.split(
            "</think>",
            1,
        )[-1].strip()

    return answer


def generate_answer(
    question: str,
    context: str,
) -> dict[str, Any]:
    """
    Виклик локальної Qwen3 через Ollama.
    """

    question = (question or "").strip()
    context = (context or "").strip()

    if not question:
        return {
            "answer": "",
            "model": MODEL_NAME,
            "latency_seconds": 0.0,
            "success": False,
            "error": "Порожнє питання.",
        }

    if not context:
        return {
            "answer": (
                "У базі знань не знайдено "
                "достатньої інформації для відповіді."
            ),
            "model": MODEL_NAME,
            "latency_seconds": 0.0,
            "success": False,
            "error": "Порожній контекст.",
        }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": _build_prompt(
                    question,
                    context,
                ),
            },
        ],
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.1,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
    }

    started = time.perf_counter()

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        latency = (
            time.perf_counter()
            - started
        )

        response.raise_for_status()

        data = response.json()

        print()
        print("OLLAMA STATS:")
        print(
            "prompt_eval_count:",
            data.get("prompt_eval_count"),
        )
        print(
            "prompt_eval_duration:",
            data.get("prompt_eval_duration"),
        )
        print(
            "eval_count:",
            data.get("eval_count"),
        )
        print(
            "eval_duration:",
            data.get("eval_duration"),
        )
        print(
            "total_duration:",
            data.get("total_duration"),
        )
        print(
            "load_duration:",
            data.get("load_duration"),
        )

        message = data.get(
            "message",
            {},
        )

        answer = message.get(
            "content",
            "",
        )

        answer = _clean_answer(answer)

        if not answer:
            return {
                "answer": "",
                "model": MODEL_NAME,
                "latency_seconds": latency,
                "success": False,
                "error": (
                    "Модель повернула "
                    "порожню відповідь."
                ),
            }

        return {
            "answer": answer,
            "model": data.get(
                "model",
                MODEL_NAME,
            ),
            "latency_seconds": latency,
            "success": True,
            "error": None,
        }

    except requests.RequestException as exc:
        latency = (
            time.perf_counter()
            - started
        )

        return {
            "answer": "",
            "model": MODEL_NAME,
            "latency_seconds": latency,
            "success": False,
            "error": (
                f"Ollama API error: {exc}"
            ),
        }


def answer_with_rag(
    question: str,
    candidates: int = 20,
    top_k: int = RAG_CONTEXT_CHUNKS,
) -> dict[str, Any]:
    """
    Повний E5 RAG pipeline:

    question
        ↓
    Hybrid BM25 + Semantic
        ↓
    RRF
        ↓
    context selection
        ↓
    compact context
        ↓
    Qwen3
        ↓
    answer + sources
    """

    started = time.perf_counter()

    retrieval = retrieve(
        question,
        candidates=candidates,
        top_k=max(
            candidates,
            top_k,
        ),
    )

    selected_retrieval = _select_rag_results(
        question,
        retrieval,
        max_chunks=top_k,
    )

    context = build_context(
        selected_retrieval,
        max_chunks=top_k,
    )

    context = _compact_context(
        context,
    )

    print()
    print("RAG CONTEXT DIAGNOSTICS:")
    print(
        "Original retrieval results:",
        len(
            retrieval.get(
                "results",
                [],
            )
        ),
    )
    print(
        "Selected RAG results:",
        len(
            selected_retrieval.get(
                "results",
                [],
            )
        ),
    )
    print(
        "Context characters:",
        len(context),
    )
    print(
        "Context words:",
        len(context.split()),
    )

    llm_result = generate_answer(
        question,
        context,
    )

    total_latency = (
        time.perf_counter()
        - started
    )

    return {
        "query": question,
        "answer": llm_result.get(
            "answer",
            "",
        ),
        "model": llm_result.get(
            "model",
            MODEL_NAME,
        ),
        "success": llm_result.get(
            "success",
            False,
        ),
        "error": llm_result.get(
            "error",
            None,
        ),
        "retrieval_method": retrieval.get(
            "method",
            "hybrid_rrf",
        ),
        "retrieval_latency_seconds": retrieval.get(
            "latency_seconds",
            0.0,
        ),
        "generation_latency_seconds": llm_result.get(
            "latency_seconds",
            0.0,
        ),
        "latency_seconds": total_latency,
        "sources": format_sources(
            selected_retrieval,
        ),
        "retrieval_results": retrieval.get(
            "results",
            [],
        ),
        "context": context,
    }


if __name__ == "__main__":
    question = "хто викладає на кафедрі"

    result = answer_with_rag(
        question,
        candidates=20,
        top_k=2,
    )

    print("=" * 80)
    print("E5 — REAL RAG PIPELINE TEST")
    print("=" * 80)

    print()
    print("QUESTION:")
    print(result["query"])

    print()
    print("METHOD:")
    print(result["retrieval_method"])

    print()
    print("MODEL:")
    print(result["model"])

    print()
    print(
        "RETRIEVAL LATENCY:",
        f"{result['retrieval_latency_seconds']:.3f}s",
    )

    print()
    print(
        "GENERATION LATENCY:",
        f"{result['generation_latency_seconds']:.3f}s",
    )

    print()
    print(
        "TOTAL LATENCY:",
        f"{result['latency_seconds']:.3f}s",
    )

    print()
    print("SUCCESS:")
    print(result["success"])

    print()
    print("ANSWER:")
    print(result["answer"])

    print()
    print("SOURCES:")

    for source in result["sources"]:
        print(
            f"- {source['source_id']} — "
            f"{source['title']}"
        )

    print()
    print("CONTEXT:")
    print("-" * 80)
    print(result["context"])

    if result["error"]:
        print()
        print("ERROR:")
        print(result["error"])
