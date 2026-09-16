import json
import re
from pathlib import Path
import numpy as np

from .ollama_client import ollama_chat, ollama_embeddings

EMBED_MODEL_DEFAULT = "nomic-embed-text"
LLM_MODEL_DEFAULT = "qwen2.5"


def _load_segments(item_dir: Path) -> list[dict]:
    tj = item_dir / "transcript.json"
    if not tj.exists():
        raise FileNotFoundError(f"Нет transcript.json: {tj} (сначала сделай транскрибацию)")
    data = json.loads(tj.read_text(encoding="utf-8"))
    return data["segments"]


def _load_rag_meta(item_dir: Path) -> dict:
    meta_path = item_dir / "rag_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_embed_model(item_dir: Path, embed_model: str | None) -> str:
    if embed_model:
        return embed_model
    meta = _load_rag_meta(item_dir)
    return meta.get("embed_model", EMBED_MODEL_DEFAULT)


def _prep(embed_model: str, kind: str, text: str) -> str:
    if "e5" in embed_model.lower():
        return f"{kind}: {text}"
    return text


def _make_chunks(segments: list[dict], max_chars: int = 1800, overlap_chars: int = 250) -> list[dict]:
    chunks = []
    cur = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return

        text = " ".join(s["text"].strip() for s in cur if s.get("text"))
        text = " ".join(text.split())
        if text:
            chunks.append({
                "start": float(cur[0]["start"]),
                "end": float(cur[-1]["end"]),
                "text": text
            })

        tail = []
        tail_len = 0
        for s in reversed(cur):
            t = (s.get("text") or "").strip()
            if not t:
                continue
            tail.insert(0, s)
            tail_len += len(t) + 1
            if tail_len >= overlap_chars:
                break

        cur = tail
        cur_len = sum(len((s.get("text") or "").strip()) + 1 for s in cur)

    for s in segments:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        if cur and (cur_len + len(t) + 1 > max_chars):
            flush()

        cur.append(s)
        cur_len += len(t) + 1

    flush()
    return chunks


def build_index(
    item_dir: Path,
    embed_model: str = EMBED_MODEL_DEFAULT,
    max_chars: int = 700,
    overlap_chars: int = 120,
) -> Path:
    item_dir = Path(item_dir).resolve()
    chunks_path = item_dir / "rag_chunks.json"
    emb_path = item_dir / "rag_emb.npy"
    meta_path = item_dir / "rag_meta.json"

    segments = _load_segments(item_dir)
    chunks = _make_chunks(segments, max_chars=max_chars, overlap_chars=overlap_chars)

    passages = [_prep(embed_model, "passage", c["text"]) for c in chunks]
    vecs = ollama_embeddings(embed_model, passages)
    X = np.asarray(vecs, dtype=np.float32)

    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(emb_path, X)
    meta_path.write_text(json.dumps({
        "embed_model": embed_model,
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "chunks": len(chunks),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return meta_path


def retrieve(
    item_dir: Path,
    question: str,
    top_k: int = 5,
    embed_model: str | None = None,
) -> list[dict]:
    item_dir = Path(item_dir).resolve()
    embed_model = _resolve_embed_model(item_dir, embed_model)

    chunks = json.loads((item_dir / "rag_chunks.json").read_text(encoding="utf-8"))
    X = np.load(item_dir / "rag_emb.npy")

    q = ollama_embeddings(embed_model, [_prep(embed_model, "query", question)])[0]
    qv = np.asarray(q, dtype=np.float32)
    qv /= (np.linalg.norm(qv) + 1e-12)

    scores = X @ qv
    idx = np.argsort(scores)[-top_k:][::-1]

    out = []
    for rank, i in enumerate(idx, start=1):
        c = chunks[int(i)]
        out.append({
            "rank": rank,
            "chunk_id": int(i),
            "score": float(scores[int(i)]),
            "start": c["start"],
            "end": c["end"],
            "text": c["text"],
        })
    return out


def _build_context_block(ctx: list[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] ({c['start']:.1f}–{c['end']:.1f} сек)\n{c['text']}"
        for i, c in enumerate(ctx)
    )


def _make_prompt(question: str, context_block: str, assistant_mode: str) -> tuple[str, str]:
    if assistant_mode == "explain":
        system = (
            "Ты учебный ассистент по лекции.\n"
            "Отвечай только по данным фрагментам лекции.\n"
            "Объясняй простыми словами, как студенту.\n"
            "Если уместно, приведи очень короткий пример."
        )
        user = (
            f"Вопрос: {question}\n\n"
            f"Фрагменты лекции:\n{context_block}\n\n"
            "Ответь просто и понятно, без лишней воды."
        )
        return system, user

    system = (
        "Ты учебный ассистент по видео/лекции.\n"
        "Отвечай только на основе 'Фрагменты лекции'.\n"
        "Нельзя добавлять знания вне контекста и нельзя делать предположения.\n"
        "Сначала дай краткий ответ, потом короткое пояснение.\n"
        "В конце добавь строку: Источники: [номера фрагментов]."
    )
    user = (
        f"Вопрос: {question}\n\n"
        f"Фрагменты лекции:\n{context_block}"
    )
    return system, user


def answer_with_rag(
    item_dir: Path,
    question: str,
    llm_model: str = LLM_MODEL_DEFAULT,
    embed_model: str | None = None,
    top_k: int = 5,
) -> dict:
    return answer_with_rag_chat(
        item_dir=item_dir,
        question=question,
        history=None,
        llm_model=llm_model,
        embed_model=embed_model,
        top_k=top_k,
        assistant_mode="answer",
    )


def answer_with_rag_chat(
    item_dir: Path,
    question: str,
    history: list[dict] | None = None,
    llm_model: str = LLM_MODEL_DEFAULT,
    embed_model: str | None = None,
    top_k: int = 5,
    assistant_mode: str = "answer",
    no_answer_threshold: float = 0.15,
) -> dict:
    ctx = retrieve(item_dir, question, top_k=top_k, embed_model=embed_model)

    if not ctx:
        return {"answer": "не упоминается в видео", "contexts": []}

    if ctx[0]["score"] < no_answer_threshold:
        return {"answer": "не упоминается в видео", "contexts": ctx}

    context_block = _build_context_block(ctx)
    system, user = _make_prompt(question, context_block, assistant_mode)

    msgs = [{"role": "system", "content": system}]
    if history:
        msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": user})

    answer = ollama_chat(model=llm_model, messages=msgs, temperature=0.2)
    return {"answer": answer, "contexts": ctx}


def _extract_json_block(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("Не удалось извлечь JSON из ответа модели")


def _sample_chunks_for_topics(item_dir: Path, max_chunks: int = 12) -> list[dict]:
    chunks_path = item_dir / "rag_chunks.json"
    if not chunks_path.exists():
        build_index(item_dir)

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if len(chunks) <= max_chunks:
        return chunks

    idxs = np.linspace(0, len(chunks) - 1, num=max_chunks, dtype=int)
    return [chunks[int(i)] for i in idxs]


def build_topics_map(
    item_dir: Path,
    llm_model: str = LLM_MODEL_DEFAULT,
    force: bool = False,
) -> dict:
    item_dir = Path(item_dir).resolve()
    topics_path = item_dir / "topics.json"

    if topics_path.exists() and not force:
        return json.loads(topics_path.read_text(encoding="utf-8"))

    sample_chunks = _sample_chunks_for_topics(item_dir)
    context = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(sample_chunks)
    )

    system = (
        "Ты анализируешь лекцию и выделяешь учебные темы.\n"
        "Верни только JSON.\n"
        "Формат:\n"
        "{\n"
        '  "topics": [\n'
        '    {"name": "...", "description": "...", "keywords": ["..."], "facets": ["общая теория","формулы","личности","определения","примеры","история"]}\n'
        "  ]\n"
        "}\n"
        "Выдели 4-8 тем. Не придумывай лишнего."
    )
    user = f"Фрагменты лекции:\n{context}"

    raw = ollama_chat(llm_model, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=0.1)

    try:
        data = _extract_json_block(raw)
        topics = data.get("topics", [])
        if not isinstance(topics, list) or not topics:
            raise ValueError("empty topics")
    except Exception:
        topics = [{
            "name": "Общий материал лекции",
            "description": "Не удалось автоматически выделить темы, используем общий охват лекции.",
            "keywords": [],
            "facets": ["общая теория"]
        }]
        data = {"topics": topics}

    topics_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def format_topics_for_chat(topics_data: dict) -> str:
    topics = topics_data.get("topics", [])
    if not topics:
        return "Не удалось выделить темы лекции."

    lines = ["Лекция охватывает такие темы:"]
    for i, t in enumerate(topics, start=1):
        facets = ", ".join(t.get("facets", []))
        desc = t.get("description", "").strip()
        line = f"{i}. {t.get('name', 'Без названия')}"
        if facets:
            line += f" — акценты: {facets}"
        if desc:
            line += f"\n   {desc}"
        lines.append(line)

    lines.append(
        "\nМожешь написать, например:\n"
        "- сделай общий тест по всей лекции\n"
        "- сделай тест полегче по формулам\n"
        "- сделай тест по личностям\n"
        "- сделай сложный тест по теме '...'"
    )
    return "\n".join(lines)


def _detect_intent(message: str) -> str:
    msg = message.lower()

    if any(x in msg for x in [
        "какие темы", "какие разделы", "что охватывает", "по каким темам",
        "темы лекции", "какие есть темы"
    ]):
        return "topics"

    if any(x in msg for x in [
        "тест", "викторин", "проверь меня", "опрос", "проверку знаний"
    ]):
        return "quiz"

    if any(x in msg for x in [
        "объясни проще", "объясни простыми словами", "для новичка", "попроще"
    ]):
        return "explain"

    return "answer"


def _fallback_quiz_request_parser(user_message: str, topics_data: dict) -> dict:
    msg = user_message.lower()

    difficulty = "medium"
    if any(x in msg for x in ["полегче", "лёгк", "легк", "простой"]):
        difficulty = "easy"
    elif any(x in msg for x in ["сложн", "посложнее", "хард"]):
        difficulty = "hard"

    focus = "all"
    if any(x in msg for x in ["формул", "уравнен", "закон"]):
        focus = "formulas"
    elif any(x in msg for x in ["личност", "учен", "учё", "автор", "кто"]):
        focus = "personalities"
    elif any(x in msg for x in ["определен", "термин", "понят"]):
        focus = "definitions"
    elif any(x in msg for x in ["пример", "применен", "задач"]):
        focus = "examples"
    elif any(x in msg for x in ["истори", "этап", "когда"]):
        focus = "history"

    questions_count = 5
    m = re.search(r"(\d+)\s*(вопрос|задан|штук)", msg)
    if m:
        questions_count = max(3, min(10, int(m.group(1))))

    selected_topics = []
    for topic in topics_data.get("topics", []):
        name = topic.get("name", "").lower()
        if name and name in msg:
            selected_topics.append(topic["name"])

    return {
        "difficulty": difficulty,
        "focus": focus,
        "topics": selected_topics,
        "questions_count": questions_count,
    }


def parse_quiz_request(user_message: str, topics_data: dict, llm_model: str) -> dict:
    topic_names = [t.get("name", "") for t in topics_data.get("topics", []) if t.get("name")]

    system = (
        "Ты парсер пользовательского запроса для учебного ассистента.\n"
        "Верни только JSON.\n"
        "Формат:\n"
        "{\n"
        '  "difficulty": "easy|medium|hard",\n'
        '  "focus": "all|formulas|personalities|definitions|examples|history",\n'
        '  "topics": ["..."],\n'
        '  "questions_count": 3\n'
        "}\n"
        "Если тема не указана явно, topics должен быть пустым списком."
    )

    user = (
        f"Доступные темы лекции: {topic_names}\n"
        f"Сообщение пользователя: {user_message}"
    )

    try:
        raw = ollama_chat(llm_model, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=0.0)
        data = _extract_json_block(raw)

        difficulty = data.get("difficulty", "medium")
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "medium"

        focus = data.get("focus", "all")
        if focus not in {"all", "formulas", "personalities", "definitions", "examples", "history"}:
            focus = "all"

        topics = data.get("topics", [])
        if not isinstance(topics, list):
            topics = []

        topic_names_set = set(topic_names)
        topics = [t for t in topics if t in topic_names_set]

        questions_count = int(data.get("questions_count", 5))
        questions_count = max(3, min(10, questions_count))

        return {
            "difficulty": difficulty,
            "focus": focus,
            "topics": topics,
            "questions_count": questions_count,
        }
    except Exception:
        return _fallback_quiz_request_parser(user_message, topics_data)


def _focus_to_human_label(focus: str) -> str:
    mapping = {
        "all": "по всем знаниям лекции",
        "formulas": "по формулам и законам",
        "personalities": "по личностям, авторам и учёным",
        "definitions": "по определениям и понятиям",
        "examples": "по примерам и применению",
        "history": "по истории и этапам развития темы",
    }
    return mapping.get(focus, "по всем знаниям лекции")


def generate_quiz_from_request(
    item_dir: Path,
    user_message: str,
    llm_model: str = LLM_MODEL_DEFAULT,
    embed_model: str | None = None,
    top_k: int = 8,
) -> dict:
    topics_data = build_topics_map(item_dir, llm_model=llm_model)
    spec = parse_quiz_request(user_message, topics_data, llm_model=llm_model)

    retrieval_parts = []

    if spec["topics"]:
        retrieval_parts.extend(spec["topics"])
    else:
        retrieval_parts.append("вся лекция")

    focus_hint = {
        "all": "главные идеи, понятия, факты, причинно-следственные связи",
        "formulas": "формулы, законы, уравнения, вычисления",
        "personalities": "личности, учёные, авторы, кто что предложил",
        "definitions": "термины, определения, ключевые понятия",
        "examples": "примеры, задачи, применение",
        "history": "история развития, этапы, хронология",
    }[spec["focus"]]

    retrieval_parts.append(focus_hint)
    retrieval_query = " ; ".join(retrieval_parts)

    ctx = retrieve(item_dir, retrieval_query, top_k=top_k, embed_model=embed_model)
    context_block = _build_context_block(ctx)

    difficulty_hint = {
        "easy": "лёгкий уровень: базовые вопросы, без подвохов",
        "medium": "средний уровень: понимание сути и связей",
        "hard": "сложный уровень: проверка глубокого понимания и нюансов",
    }[spec["difficulty"]]

    topics_text = ", ".join(spec["topics"]) if spec["topics"] else "вся лекция"
    focus_text = _focus_to_human_label(spec["focus"])

    system = (
        "Ты учебный ассистент.\n"
        "Составь тест только по предоставленным фрагментам лекции.\n"
        "Не добавляй ничего вне контекста.\n"
        "Формат:\n"
        "1. Вопрос\n"
        "   A) ...\n"
        "   B) ...\n"
        "   C) ...\n"
        "   D) ...\n"
        "   Правильный ответ: ...\n"
        "   Короткое объяснение: ...\n"
    )

    user = (
        f"Сделай тест.\n"
        f"Охват: {topics_text}\n"
        f"Фокус: {focus_text}\n"
        f"Сложность: {difficulty_hint}\n"
        f"Количество вопросов: {spec['questions_count']}\n\n"
        f"Фрагменты лекции:\n{context_block}"
    )

    answer = ollama_chat(model=llm_model, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=0.3)

    prefix = (
        f"Сделал тест: {spec['questions_count']} вопросов, "
        f"{'по темам ' + topics_text if spec['topics'] else 'по всей лекции'}, "
        f"{focus_text}, уровень — {spec['difficulty']}.\n\n"
    )

    return {
        "answer": prefix + answer.strip(),
        "contexts": ctx,
        "quiz_spec": spec,
    }


def assistant_reply(
    item_dir: Path,
    user_message: str,
    history: list[dict] | None = None,
    llm_model: str = LLM_MODEL_DEFAULT,
    embed_model: str | None = None,
    top_k: int = 5,
) -> dict:
    intent = _detect_intent(user_message)

    if intent == "topics":
        topics_data = build_topics_map(item_dir, llm_model=llm_model)
        return {
            "answer": format_topics_for_chat(topics_data),
            "contexts": [],
        }

    if intent == "quiz":
        return generate_quiz_from_request(
            item_dir=item_dir,
            user_message=user_message,
            llm_model=llm_model,
            embed_model=embed_model,
            top_k=max(top_k, 7),
        )

    if intent == "explain":
        return answer_with_rag_chat(
            item_dir=item_dir,
            question=user_message,
            history=history,
            llm_model=llm_model,
            embed_model=embed_model,
            top_k=top_k,
            assistant_mode="explain",
        )

    return answer_with_rag_chat(
        item_dir=item_dir,
        question=user_message,
        history=history,
        llm_model=llm_model,
        embed_model=embed_model,
        top_k=top_k,
        assistant_mode="answer",
    )