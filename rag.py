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


SYSTEM_PROMPT = """
Ти — інформаційний асистент кафедри систем
штучного інтелекту та аналізу даних ТНТУ.

Відповідай тільки на основі переданого контексту.
Не вигадуй факти, дати, адреси, телефони,
електронні адреси, освітні програми, прізвища
викладачів або умови вступу.

Якщо контексту недостатньо — прямо скажи про це.
Відповідай українською мовою.
Не показуй процес міркування.

Для використаних джерел використовуй тільки номери
у форматі [1], [2], [3], які відповідають КОНТЕКСТУ.
Не створюй вигаданих джерел.
"""


def _remove_urls_from_context(context: str) -> str:
    """
    Прибирає URL із тексту контексту перед передачею LLM.

    URL залишаються в окремому полі sources і тому
    не втрачаються для frontend.
    """

    if not context:
        return ""

    # Приклад:
    # | url=https://kaf-ai.tntu.edu.ua/news/
    context = re.sub(
        r"\s*\|\s*url=https?://\S+",
        "",
        context,
        flags=re.IGNORECASE,
    )

    # Додатково прибираємо окремі рядки URL,
    # якщо вони трапляються у контексті.
    context = re.sub(
        r"^\s*https?://\S+\s*$",
        "",
        context,
        flags=re.MULTILINE,
    )

    return context.strip()


def _build_prompt(
    question: str,
    context: str,
) -> str:
    return f"""
КОНТЕКСТ БАЗИ ЗНАНЬ
===================
{context}
===================

ПИТАННЯ
===================
{question}
===================

Сформулюй коротку конкретну відповідь українською.
Використовуй тільки наведений контекст.
Не вигадуй інформацію.
Посилання на джерела подавай у форматі [1], [2], [3].
"""


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
                "достатньої інформації "
                "для відповіді."
            ),
            "model": MODEL_NAME,
            "latency_seconds": 0.0,
            "success": False,
            "error": "Порожній контекст.",
        }

    context = _remove_urls_from_context(context)

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
            "num_predict": 128,
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

        message = data.get(
            "message",
            {},
        )

        answer = message.get(
            "content",
            "",
        )

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
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Повний E5 pipeline:

    question
       ↓
    Hybrid BM25 + Semantic
       ↓
    RRF
       ↓
    TOP-K
       ↓
    context
       ↓
    Qwen3
       ↓
    answer + sources
    """

    started = time.perf_counter()

    retrieval = retrieve(
        question,
        candidates=candidates,
        top_k=top_k,
    )

    context = build_context(
        retrieval,
        max_chunks=top_k,
    )

    context = _remove_urls_from_context(
        context
    )

    print()
    print("CONTEXT DIAGNOSTICS:")
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
            "error"
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
            retrieval
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
        top_k=3,
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
