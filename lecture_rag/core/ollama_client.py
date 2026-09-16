import time
import requests

OLLAMA_URL = "http://localhost:11434"


def ollama_chat(model: str, messages: list[dict], temperature: float = 0.2) -> str:
    """
    Локальный чат через Ollama: POST /api/chat
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"Ollama chat error {r.status_code}: {r.text}")
    return r.json()["message"]["content"]


def ollama_embeddings(model: str, texts: list[str], batch_size: int = 1) -> list[list[float]]:
    """
    Эмбеддинги через Ollama: POST /api/embed

    На Windows у Ollama бывают падения при input=list (много фрагментов за раз),
    поэтому по умолчанию делаем батч 1 и если 1 элемент — шлём строкой, не массивом.
    Плюс truncate=True, и несколько попыток (runner может перезапускаться).
    """
    out: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        payload = {
            "model": model,
            "input": batch[0] if len(batch) == 1 else batch,
            "truncate": True,
        }

        for attempt in range(3):
            r = requests.post(f"{OLLAMA_URL}/api/embed", json=payload, timeout=600)

            if r.status_code < 400:
                data = r.json()
                embs = data["embeddings"]
                if len(batch) == 1:
                    out.append(embs[0])
                else:
                    out.extend(embs)
                break

            # если runner “упал” — подождать и повторить
            txt = r.text.lower()
            if "forcibly closed" in txt or "wsarecv" in txt:
                time.sleep(1.0 * (attempt + 1))
                continue

            raise RuntimeError(f"Ollama embed error {r.status_code}: {r.text}")

        else:
            raise RuntimeError(f"Ollama embed error: runner keeps failing on batch starting at {i}")

    return out