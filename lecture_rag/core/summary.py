import json
from pathlib import Path
from datetime import datetime

from .ollama_client import ollama_chat


LENGTH_PRESETS = {
    "Коротко": {"chunk_bullets": 3, "final_bullets": 8, "final_chars_hint": 1200},
    "Средне": {"chunk_bullets": 5, "final_bullets": 15, "final_chars_hint": 2500},
    "Подробно": {"chunk_bullets": 8, "final_bullets": 25, "final_chars_hint": 4500},
}


def _load_segments(item_dir: Path) -> list[dict]:
    tj = item_dir / "transcript.json"
    if not tj.exists():
        raise FileNotFoundError(f"Нет transcript.json: {tj} (сначала сделай транскрибацию)")
    data = json.loads(tj.read_text(encoding="utf-8"))
    return data["segments"]


def _make_chunks_from_segments(segments: list[dict], max_chars: int = 1200, overlap_chars: int = 150) -> list[dict]:
    """
    Склеиваем сегменты Whisper в чанки по символам.
    """
    chunks = []
    cur = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        text = " ".join((s.get("text") or "").strip() for s in cur if s.get("text"))
        text = " ".join(text.split())
        if text:
            chunks.append({
                "start": float(cur[0]["start"]),
                "end": float(cur[-1]["end"]),
                "text": text
            })

        # overlap
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


def _get_chunks(item_dir: Path) -> list[dict]:
    # Для сводки режем заново без overlap, иначе будут повторы из rag_chunks
    segments = _load_segments(item_dir)
    return _make_chunks_from_segments(segments, max_chars=1400, overlap_chars=0)


def summarize_video(
    item_dir: Path,
    llm_model: str = "qwen2.5",
    length_mode: str = "Средне",
    bullet_style: bool = True,
    remove_filler: bool = True,
    keep_structure: bool = True,
) -> dict:
    """
    Иерархическая сводка:
      1) кратко суммируем каждый чанк
      2) суммируем все мини-сводки в итоговую

    Возвращает dict: {"summary": str, "meta": {...}}
    Сохранять на диск будет окно (summary_window).
    """
    item_dir = Path(item_dir).resolve()
    if length_mode not in LENGTH_PRESETS:
        length_mode = "Средне"

    preset = LENGTH_PRESETS[length_mode]
    chunk_bullets = preset["chunk_bullets"]
    final_bullets = preset["final_bullets"]
    final_chars_hint = preset["final_chars_hint"]

    chunks = _get_chunks(item_dir)
    if not chunks:
        raise RuntimeError("Не найдено текста для сводки (пустые chunks).")

    system = (
        "Ты редактор научпоп-конспекта.\n"
        "Пиши по-русски, аккуратно и кратко.\n"
        "Запрещено повторяться: одинаковые мысли объединяй.\n"
        "Запрещено добавлять знания не из текста.\n"
        "Не делай 'пункт про каждое предложение'.\n"
        "Цель — человеческая сводка, а не расшифровка."
    )

    # 1) мини-сводки по чанкам
    mini = []
    for i, c in enumerate(chunks, start=1):
        style = "в виде маркеров" if bullet_style else "в 2-3 абзацах"
        extra = []
        if remove_filler:
            extra.append("Убери слова-паразиты, повторы, лишние вводные фразы.")
        if keep_structure:
            extra.append("Сохраняй смысловую структуру/темы, если она видна.")

        user = (
            f"Выдели из фрагмента только СМЫСЛ и НОВЫЕ факты.\n"
            f"Формат:\n"
            f"- Идеи: максимум {chunk_bullets} коротких пунктов (без воды)\n"
            f"- Факты/числа/имена: 0–3 пункта (только если есть)\n"
            f"Если во фрагменте нет ничего нового — напиши: 'нет новых идей'.\n"
            f"Текст:\n{c['text']}"
        )
        out = ollama_chat(llm_model, [{"role": "system", "content": system},
                                      {"role": "user", "content": user}],
                          temperature=0.2)
        mini.append(out.strip())

    # 2) итоговая сводка из мини-сводок
    joined = "\n\n".join(mini)
    final_style = "маркированный список" if bullet_style else "короткий связный текст"
    final_user = (
        "Собери итоговую сводку по всей лекции на основе мини-сводок ниже.\n"
        "ЖЁСТКО: не повторяйся, дубли объединяй, лишнее выкидывай.\n"
        "Не добавляй ничего от себя.\n\n"
        "Сделай результат в таком виде:\n"
        "TL;DR: 2–3 предложения.\n"
        "Ключевые идеи: 6–10 пунктов.\n"
        "Персонажи/вклад: 2–5 пунктов (кто что объяснил).\n"
        "Числа и факты: отдельный список (только то, что явно есть в тексте).\n\n"
        "Мини-сводки:\n"
        f"{joined}"
    )
    final_summary = ollama_chat(llm_model, [{"role": "system", "content": system},
                                            {"role": "user", "content": final_user}],
                                temperature=0.2).strip()

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "llm_model": llm_model,
        "length_mode": length_mode,
        "bullet_style": bullet_style,
        "remove_filler": remove_filler,
        "keep_structure": keep_structure,
        "chunks_used": len(chunks),
    }

    return {"summary": final_summary, "meta": meta}