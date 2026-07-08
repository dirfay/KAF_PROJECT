import os
import json
import time
from pathlib import Path
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from openai import OpenAI

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "site_content.json"
COLLECTION_NAME = "kaf_content"
BATCH_SIZE = 10
SLEEP_BETWEEN = 0.35

if not OPENAI_KEY:
    raise SystemExit("ERROR: встанови OPENAI_API_KEY у змінних оточення.")

openai_client = OpenAI(api_key=OPENAI_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL)


def get_embedding(text: str) -> List[float]:
    """
    Використовує новий openai client API (openai>=1.0.0).
    Повертає вектор для переданого тексту.
    """
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


def load_docs():
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing {DATA_PATH} — створіть файл з контентом.")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    if data.get("announcement"):
        docs.append({"id": "announcement", "title": "Announcement",
                    "text": data["announcement"], "meta": {"type": "announcement"}})
    for i, n in enumerate(data.get("news", [])):
        text = (n.get("title", "") + "\n\n" + n.get("body", "")).strip()
        docs.append({"id": f"news-{i}", "title": n.get("title", ""), "text": text,
                    "meta": {"type": "news", "date": n.get("date"), "slug": n.get("slug")}})
    for i, f in enumerate(data.get("faq", [])):
        text = (f.get("q", "") + "\n\n" + f.get("a", "")).strip()
        docs.append({"id": f"faq-{i}", "title": f.get("q", ""),
                    "text": text, "meta": {"type": "faq"}})
    return docs


def ensure_collection_with_dynamic_size(example_text: str):
    """
    Якщо колекції немає — отримаємо embedding прикладу, визначимо розмір вектору
    і створимо колекцію з відповідним VectorParams.
    """
    exists = qdrant_client.collection_exists(collection_name=COLLECTION_NAME)
    if exists:
        print("Collection exists:", COLLECTION_NAME)
        return

    print("Collection not found — робимо перший embedding щоб визначити розмір вектору...")
    emb = get_embedding(example_text)
    vec_size = len(emb)
    print("Detected embedding size:", vec_size)
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(
            size=vec_size, distance=rest.Distance.COSINE)
    )
    print("Collection created:", COLLECTION_NAME)


def upsert_docs(docs):
    total = len(docs)
    print(f"Upserting {total} documents in batches of {BATCH_SIZE}...")
    idx = 0
    while idx < total:
        batch = []
        for d in docs[idx: idx + BATCH_SIZE]:
            try:
                emb = get_embedding(d["text"])
            except Exception as e:
                print("OpenAI embedding error for doc", d["id"], "->", e)
                raise
            payload = {
                "title": d["title"],
                "meta": d["meta"],
                "text": d["text"]
            }
            point = rest.PointStruct(id=d["id"], vector=emb, payload=payload)
            batch.append(point)
            time.sleep(SLEEP_BETWEEN)
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch)
        idx += BATCH_SIZE
        print(f"Upserted {min(idx, total)}/{total} points")
    print("All upsert done.")


def main():
    docs = load_docs()
    print("Docs to index:", len(docs))
    if not docs:
        print("No documents found to index.")
        return

    ensure_collection_with_dynamic_size(docs[0]["text"])
    upsert_docs(docs)
    print("Ingest finished.")


if __name__ == "__main__":
    main()
