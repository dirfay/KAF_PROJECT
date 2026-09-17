from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag import answer_with_rag


router = APIRouter()


class AskRagPayload(BaseModel):
    question: str
    url: str | None = None
    title: str | None = None


@router.post("/ask-rag")
def ask_rag(payload: AskRagPayload):
    question = (payload.question or "").strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Питання не може бути порожнім."
        )

    try:
        result = answer_with_rag(
            question,
            candidates=20,
            top_k=5
        )
    except Exception as exc:
        return {
            "success": False,
            "answer": "Не вдалося отримати відповідь від RAG-асистента.",
            "source": "error",
            "error": str(exc)
        }

    if not result.get("success"):
        return {
            "success": False,
            "answer": "Не вдалося сформувати відповідь.",
            "source": "error",
            "error": result.get("error", "Невідома помилка.")
        }

    sources = result.get("sources", [])

    source_titles = []
    for source in sources:
        title = source.get("title") or source.get("source_id")
        if title and title not in source_titles:
            source_titles.append(title)

    source_text = ", ".join(source_titles)

    return {
        "success": True,
        "answer": result.get("answer", ""),
        "source": source_text,
        "sources": sources,
        "model": result.get("model"),
        "retrieval_method": result.get("retrieval_method"),
        "retrieval_latency": result.get("retrieval_latency_seconds"),
        "generation_latency": result.get("generation_latency_seconds"),
        "latency": result.get("latency_seconds")
    }
