from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, Request, Form, HTTPException, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pathlib import Path
from datetime import datetime
import json
import os
import secrets
import shutil


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "site_content.json"
QUESTION_LOGS_PATH = BASE_DIR.parent / "data" / "question_logs.json"

app = FastAPI(title="KAF Project Demo")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

security = HTTPBasic()
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")


def check_admin(creds: HTTPBasicCredentials = Depends(security)):
    user_ok = secrets.compare_digest(creds.username or "", ADMIN_USER)
    pass_ok = secrets.compare_digest(creds.password or "", ADMIN_PASS)
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


def ensure_data_folder():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def ensure_logs_file():
    ensure_data_folder()
    if not QUESTION_LOGS_PATH.exists():
        with open(QUESTION_LOGS_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_site_content():
    ensure_data_folder()
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        initial = {
            "announcement": "Набір на нову магістерську програму з ШІ відкрито. Деталі — у розділі «Освітні програми».",
            "news": [
                {
                    "date": "2025-10-01",
                    "title": "Відкрито набір на магістратуру",
                    "slug": "magistr-nabor",
                    "body": "Деталі про програму і вимоги до вступу..."
                },
                {
                    "date": "2025-09-20",
                    "title": "Участь у конференції",
                    "slug": "conf-2025",
                    "body": "Наші викладачі представили результати на міжнародній конференції."
                }
            ],
            "contacts": {
                "department_name": "Кафедра систем Штучного Інтелекту та Аналізу Даних",
                "university_name": "Тернопільський національний технічний університет імені Івана Пулюя",
                "address": "ТНТУ, вул. Руська, 56, корп.2, к.62",
                "phone": "+380931991085",
                "email": "kaf_ai@tntu.edu.ua",
                "office_hours": "Понеділок — П'ятниця: 10:00 — 12:00",
                "keywords": ["контакти", "адреса", "телефон", "email", "пошта", "де знаходиться", "де кафедра"]
            },
            "faq": [],
            "programs": [],
            "admission": {},
            "internships": {},
            "science": {},
            "assistant": {
                "welcome_title": "Вітаю! Я асистент кафедри.",
                "welcome_text": "Я можу допомогти з питаннями про вступ, освітні програми, контакти, графік прийому, стажування, наукову діяльність та новини кафедри.",
                "example_questions": [
                    "Як вступити?",
                    "Де знаходиться кафедра?",
                    "Які є освітні програми?"
                ],
                "fallback_answer": "Я поки не знайшов точної відповіді у локальній базі сайту. Спробуйте уточнити запит або перегляньте розділи «Абітурієнту», «Освітні програми» чи «Контакти»."
            }
        }
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(initial, f, ensure_ascii=False, indent=2)
        return initial


def backup_content():
    if DATA_PATH.exists():
        bak = DATA_PATH.with_suffix(DATA_PATH.suffix + ".bak")
        shutil.copy2(DATA_PATH, bak)


def render_template(name: str, request: Request, **ctx):
    ctx.setdefault("current_year", datetime.now().year)
    ctx.setdefault("content", load_site_content())
    ctx.setdefault("request", request)
    return templates.TemplateResponse(name, ctx)


def load_question_logs():
    ensure_logs_file()
    try:
        with open(QUESTION_LOGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def save_question_log(question: str, url: Optional[str], title: Optional[str], source: str, status: str):
    logs = load_question_logs()
    logs.append(
        {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "url": url or "",
            "title": title or "",
            "source": source,
            "status": status
        }
    )
    with open(QUESTION_LOGS_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(normalize_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(normalize_text(v) for v in value.values())
    return str(value)


def tokenize_question(question: str):
    cleaned = (
        question.lower()
        .replace("?", " ")
        .replace("!", " ")
        .replace(".", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("«", " ")
        .replace("»", " ")
        .replace('"', " ")
        .replace("'", " ")
    )
    words = [w.strip() for w in cleaned.split() if len(w.strip()) > 2]
    return words


def contains_any_keyword(question: str, keywords):
    if not keywords:
        return False
    q = question.lower()
    for keyword in keywords:
        if str(keyword).lower() in q:
            return True
    return False


def question_matches_text(question: str, text: str):
    words = tokenize_question(question)
    text_lower = text.lower()
    if not words:
        return False
    matches = sum(1 for w in words if w in text_lower)
    return matches > 0


def search_faq(question: str, faq_list):
    q_lower = question.lower()
    for entry in faq_list:
        faq_q = str(entry.get("q", "")).strip().lower()
        faq_a = str(entry.get("a", "")).strip()
        if not faq_q or not faq_a:
            continue
        if faq_q == q_lower:
            return faq_a, "faq"
        if faq_q in q_lower:
            return faq_a, "faq"
        if question_matches_text(question, faq_q):
            return faq_a, "faq"
    return None, None


def search_contacts(question: str, contacts):
    if not isinstance(contacts, dict):
        return None, None

    keywords = contacts.get("keywords", [])
    contacts_text = normalize_text(contacts)

    if contains_any_keyword(question, keywords) or question_matches_text(question, contacts_text):
        address = contacts.get("address", "Адресу не вказано.")
        phone = contacts.get("phone", "Телефон не вказано.")
        email = contacts.get("email", "Email не вказано.")
        office_hours = contacts.get("office_hours", "Графік не вказано.")
        answer = (
            f"Контакти кафедри:\n"
            f"Адреса: {address}\n"
            f"Телефон: {phone}\n"
            f"Email: {email}\n"
            f"Графік прийому: {office_hours}"
        )
        return answer, "contacts"

    return None, None


def search_programs(question: str, programs):
    if not isinstance(programs, list):
        return None, None

    for program in programs:
        program_text = normalize_text(program)
        keywords = program.get("keywords", [])
        if contains_any_keyword(question, keywords) or question_matches_text(question, program_text):
            level = program.get("level", "Рівень не вказано")
            title = program.get("title", "Назву не вказано")
            description = program.get("description", "Опис відсутній")
            answer = (
                f"Освітня програма:\n"
                f"Рівень: {level}\n"
                f"Назва: {title}\n"
                f"Опис: {description}"
            )
            return answer, f"programs: {title}"

    return None, None


def search_section_dict(question: str, section_data: dict, section_name: str, default_title: str):
    if not isinstance(section_data, dict) or not section_data:
        return None, None

    keywords = section_data.get("keywords", [])
    section_text = normalize_text(section_data)

    if contains_any_keyword(question, keywords) or question_matches_text(question, section_text):
        summary = section_data.get("summary", "")
        details = section_data.get("details", "")
        how_to_apply = normalize_text(section_data.get("how_to_apply", ""))
        documents = normalize_text(section_data.get("documents", ""))
        entry_exams = normalize_text(section_data.get("entry_exams", ""))
        tuition = normalize_text(section_data.get("tuition", ""))
        dormitory = normalize_text(section_data.get("dormitory", ""))
        partners = normalize_text(section_data.get("partners", ""))
        activities = normalize_text(section_data.get("activities", ""))

        parts = []
        if summary:
            parts.append(summary)
        if details:
            parts.append(details)
        if how_to_apply:
            parts.append(f"Як податися: {how_to_apply}")
        if documents:
            parts.append(f"Документи: {documents}")
        if entry_exams:
            parts.append(f"Вступні вимоги: {entry_exams}")
        if tuition:
            parts.append(f"Навчання: {tuition}")
        if dormitory:
            parts.append(f"Гуртожиток: {dormitory}")
        if partners:
            parts.append(f"Партнери: {partners}")
        if activities:
            parts.append(f"Напрями або активності: {activities}")

        answer = "\n".join([p for p in parts if p]).strip()
        if not answer:
            answer = default_title

        return answer, section_name

    return None, None


def search_news(question: str, news_list):
    if not isinstance(news_list, list):
        return None, None

    for item in news_list:
        item_text = normalize_text(item)
        if question_matches_text(question, item_text):
            title = item.get("title", "Новина")
            body = item.get("body", "Опис відсутній")
            date = item.get("date", "Дата не вказана")
            answer = f"Новина від {date}: {title}\n{body}"
            return answer, f"news: {title}"

    return None, None


@app.get("/", response_class=HTMLResponse, name="home")
async def home(request: Request):
    return render_template("home.html", request)


@app.get("/contacts", response_class=HTMLResponse, name="contacts")
async def contacts(request: Request):
    return render_template("contacts.html", request)


@app.post("/contacts", name="post_contacts")
async def post_contacts(
    request: Request,
    name: str = Form(None),
    email: str = Form(None),
    subject: str = Form(None),
    message: str = Form(None)
):
    ensure_data_folder()
    submissions_path = DATA_PATH.parent / "contact_submissions.json"
    submissions = []

    try:
        if submissions_path.exists():
            with open(submissions_path, "r", encoding="utf-8") as f:
                submissions = json.load(f)
    except Exception:
        submissions = []

    submission = {
        "name": name or "",
        "email": email or "",
        "subject": subject or "",
        "message": message or "",
        "timestamp": datetime.now().isoformat()
    }

    submissions.append(submission)

    with open(submissions_path, "w", encoding="utf-8") as f:
        json.dump(submissions, f, ensure_ascii=False, indent=2)

    return render_template("contacts.html", request, saved=True, form_data=submission)


@app.get("/news", response_class=HTMLResponse, name="news")
async def news(request: Request, q: str = Query("", alias="q")):
    content = load_site_content()
    news_list = content.get("news", [])

    if q:
        ql = q.lower()
        news_list = [
            n for n in news_list
            if ql in n.get("title", "").lower() or ql in n.get("body", "").lower()
        ]

    return templates.TemplateResponse(
        "news.html",
        {
            "request": request,
            "content": content,
            "news_list": news_list,
            "query": q
        }
    )


@app.get("/programs", response_class=HTMLResponse, name="programs")
async def programs(request: Request):
    return render_template("programs.html", request)


@app.get("/abiturients", response_class=HTMLResponse, name="abiturients")
async def abiturients(request: Request):
    return render_template("abiturients.html", request)


@app.get("/science", response_class=HTMLResponse, name="science")
async def science(request: Request):
    return render_template("science.html", request)


@app.get("/schedule", response_class=HTMLResponse, name="schedule")
async def schedule(request: Request):
    return render_template("schedule.html", request)


@app.get("/internships", response_class=HTMLResponse, name="internships")
async def internships(request: Request):
    return render_template("internships.html", request)


@app.get("/search", response_class=HTMLResponse, name="search")
async def search(request: Request, q: str = Query("", alias="q")):
    return await news(request, q)


@app.get("/admin", response_class=HTMLResponse, name="admin")
def admin_page(request: Request, ok: bool = Depends(check_admin)):
    content = load_site_content()
    pretty = json.dumps(content, ensure_ascii=False, indent=2)
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "content_json": pretty,
            "saved": False
        }
    )


@app.post("/admin/save", response_class=HTMLResponse, name="admin_save")
def admin_save(request: Request, json_text: str = Form(...), ok: bool = Depends(check_admin)):
    try:
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            raise ValueError("Root JSON must be an object (dictionary).")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    ensure_data_folder()
    backup_content()

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "content_json": pretty,
            "saved": True
        }
    )


class AskPayload(BaseModel):
    question: str
    url: Optional[str] = None
    title: Optional[str] = None


@app.post("/ask", response_class=JSONResponse, name="ask")
async def ask_endpoint(payload: AskPayload):
    q = (payload.question or "").strip()
    if not q:
        return {
            "answer": "Напишіть, будь ласка, коротке питання.",
            "source": "validation"
        }

    content = load_site_content()

    answer, source = search_faq(q, content.get("faq", []))
    if answer:
        save_question_log(q, payload.url, payload.title, source, "found")
        return {"answer": answer, "source": source}

    answer, source = search_contacts(q, content.get("contacts", {}))
    if answer:
        save_question_log(q, payload.url, payload.title, source, "found")
        return {"answer": answer, "source": source}

    answer, source = search_programs(q, content.get("programs", []))
    if answer:
        save_question_log(q, payload.url, payload.title, source, "found")
        return {"answer": answer, "source": source}

    answer, source = search_section_dict(
        q,
        content.get("admission", {}),
        "admission",
        "Інформація про вступ доступна у розділі «Абітурієнту»."
    )
    if answer:
        save_question_log(q, payload.url, payload.title, source, "found")
        return {"answer": answer, "source": source}

    answer, source = search_section_dict(
        q,
        content.get("internships", {}),
        "internships",
        "Інформація про стажування доступна у розділі «Практики та стажування»."
    )
    if answer:
        save_question_log(q, payload.url, payload.title, source, "found")
        return {"answer": answer, "source": source}

    answer, source = search_section_dict(
        q,
        content.get("science", {}),
        "science",
        "Інформація про наукову діяльність доступна у розділі «Наука»."
    )
    if answer:
        save_question_log(q, payload.url, payload.title, source, "found")
        return {"answer": answer, "source": source}

    answer, source = search_news(q, content.get("news", []))
    if answer:
        save_question_log(q, payload.url, payload.title, source, "found")
        return {"answer": answer, "source": source}

    fallback_answer = (
        content.get("assistant", {}).get(
            "fallback_answer",
            "Я поки не знайшов точної відповіді у локальній базі сайту. Спробуйте уточнити запит."
        )
    )

    save_question_log(q, payload.url, payload.title, "fallback", "not_found")
    return {"answer": fallback_answer, "source": "fallback"}
