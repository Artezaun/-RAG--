from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QThread, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QLineEdit,
    QPushButton, QComboBox, QSpinBox, QGroupBox, QMessageBox, QCheckBox
)
from shiboken6 import isValid

from core.rag import build_index, answer_with_rag_chat
from core.voice import record_wav, transcribe_question, speak_text


@dataclass
class RagResult:
    answer: str
    contexts: list[dict]
    mode: str


@dataclass
class VoiceAskResult:
    question: str
    answer: str
    contexts: list[dict]
    mode: str


class RagWorker(QObject):
    finished = Signal(object)   # RagResult
    error = Signal(str)

    def __init__(
        self,
        item_dir: Path,
        question: str,
        history: list[dict],
        llm_model: str,
        top_k: int,
        rebuild: bool,
        assistant_mode: str,
        quiz_questions: int,
    ):
        super().__init__()
        self.item_dir = Path(item_dir).resolve()
        self.question = question
        self.history = history
        self.llm_model = llm_model
        self.top_k = top_k
        self.rebuild = rebuild
        self.assistant_mode = assistant_mode
        self.quiz_questions = quiz_questions

    def _make_effective_question(self) -> str:
        question = self.question.strip()

        if self.assistant_mode == "explain":
            if not question:
                return "Объясни простыми словами основную идею лекции."
            return f"Объясни простыми словами, как студенту-новичку: {question}"

        if self.assistant_mode == "quiz":
            if not question:
                question = "Сделай мини-тест по лекции."
            return (
                f"{question}\n\n"
                f"Составь мини-тест из {self.quiz_questions} вопросов только по фрагментам лекции. "
                f"Для каждого вопроса дай 4 варианта ответа, укажи правильный ответ и короткое объяснение."
            )

        return question

    def run(self):
        try:
            meta = self.item_dir / "rag_meta.json"
            if self.rebuild or not meta.exists():
                build_index(self.item_dir)

            effective_question = self._make_effective_question()
            res = answer_with_rag_chat(
                self.item_dir,
                effective_question,
                history=self.history,
                llm_model=self.llm_model,
                top_k=self.top_k,
            )

            self.finished.emit(RagResult(res["answer"], res["contexts"], self.assistant_mode))
        except Exception as e:
            self.error.emit(str(e))


class VoiceAskWorker(QObject):
    finished = Signal(object)   # VoiceAskResult
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        item_dir: Path,
        history: list[dict],
        llm_model: str,
        top_k: int,
        rebuild: bool,
        assistant_mode: str,
        quiz_questions: int,
        mic_seconds: int = 6,
        stt_model: str = "small",
        stt_device: str = "cpu",
        stt_lang: str | None = None,
    ):
        super().__init__()
        self.item_dir = Path(item_dir).resolve()
        self.history = history
        self.llm_model = llm_model
        self.top_k = top_k
        self.rebuild = rebuild
        self.assistant_mode = assistant_mode
        self.quiz_questions = quiz_questions
        self.mic_seconds = mic_seconds
        self.stt_model = stt_model
        self.stt_device = stt_device
        self.stt_lang = stt_lang

    def _make_effective_question(self, question: str) -> str:
        question = question.strip()

        if self.assistant_mode == "explain":
            if not question:
                return "Объясни простыми словами основную идею лекции."
            return f"Объясни простыми словами, как студенту-новичку: {question}"

        if self.assistant_mode == "quiz":
            if not question:
                question = "Сделай мини-тест по лекции."
            return (
                f"{question}\n\n"
                f"Составь мини-тест из {self.quiz_questions} вопросов только по фрагментам лекции. "
                f"Для каждого вопроса дай 4 варианта ответа, укажи правильный ответ и короткое объяснение."
            )

        return question

    def run(self):
        try:
            audio_path = self.item_dir / "_last_voice_question.wav"

            self.progress.emit(f"Слушаю микрофон… {self.mic_seconds} сек")
            record_wav(audio_path, seconds=float(self.mic_seconds))

            self.progress.emit("Распознаю вопрос…")
            spoken_question = transcribe_question(
                audio_path,
                model_name=self.stt_model,
                device=self.stt_device,
                lang=self.stt_lang,
            )

            if not spoken_question.strip():
                raise RuntimeError("Не удалось распознать вопрос с микрофона.")

            meta = self.item_dir / "rag_meta.json"
            if self.rebuild or not meta.exists():
                build_index(self.item_dir)

            effective_question = self._make_effective_question(spoken_question)

            self.progress.emit("Формирую ответ…")
            res = answer_with_rag_chat(
                self.item_dir,
                effective_question,
                history=self.history,
                llm_model=self.llm_model,
                top_k=self.top_k,
            )

            self.finished.emit(
                VoiceAskResult(
                    question=spoken_question,
                    answer=res["answer"],
                    contexts=res["contexts"],
                    mode=self.assistant_mode,
                )
            )
        except Exception as e:
            self.error.emit(str(e))


class TtsWorker(QObject):
    finished = Signal()
    error = Signal(str)

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def run(self):
        try:
            speak_text(self.text)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class ChatWindow(QWidget):
    def __init__(self, item_dir: Path):
        super().__init__()
        self.setWindowTitle("RAG чат по видео")
        self.resize(900, 650)

        self.item_dir = Path(item_dir).resolve()
        self.history: list[dict] = []
        self.chat_path = self.item_dir / "chat_history.json"

        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._rebuild_next = False

        self._tts_thread: QThread | None = None
        self._tts_worker: TtsWorker | None = None

        layout = QVBoxLayout(self)

        # --- controls ---
        box = QGroupBox("Настройки")
        row = QHBoxLayout(box)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["qwen2.5", "qwen2.5:1.5b", "qwen2.5:7b"])
        self.model_combo.setCurrentText("qwen2.5")

        self.topk = QSpinBox()
        self.topk.setRange(1, 12)
        self.topk.setValue(5)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Ответ по лекции", "answer")
        self.mode_combo.addItem("Объяснить проще", "explain")
        self.mode_combo.addItem("Мини-тест", "quiz")

        self.quiz_count = QSpinBox()
        self.quiz_count.setRange(3, 10)
        self.quiz_count.setValue(5)

        self.mic_seconds = QSpinBox()
        self.mic_seconds.setRange(2, 20)
        self.mic_seconds.setValue(6)

        self.btn_voice = QPushButton("🎤 Голосовой вопрос")
        self.btn_voice.clicked.connect(self.on_voice_ask)

        self.chk_voice_reply = QCheckBox("Озвучивать ответ")
        self.chk_voice_reply.setChecked(True)

        self.btn_rebuild = QPushButton("Пересобрать индекс (следующий запрос)")
        self.btn_rebuild.clicked.connect(self.mark_rebuild)

        self.btn_save_chat = QPushButton("Сохранить чат")
        self.btn_delete_chat = QPushButton("Удалить чат")
        self.btn_save_chat.clicked.connect(self.save_chat)
        self.btn_delete_chat.clicked.connect(self.delete_chat)

        row.addWidget(QLabel("LLM модель:"))
        row.addWidget(self.model_combo, 1)
        row.addWidget(QLabel("top_k:"))
        row.addWidget(self.topk)
        row.addWidget(QLabel("Режим:"))
        row.addWidget(self.mode_combo)
        row.addWidget(QLabel("Вопросов в тесте:"))
        row.addWidget(self.quiz_count)
        row.addWidget(QLabel("Запись, сек:"))
        row.addWidget(self.mic_seconds)
        row.addWidget(self.btn_voice)
        row.addWidget(self.chk_voice_reply)
        row.addWidget(self.btn_rebuild)
        row.addWidget(self.btn_save_chat)
        row.addWidget(self.btn_delete_chat)

        # --- chat view ---
        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setPlaceholderText("Здесь будет диалог…")

        # --- sources view ---
        self.sources = QTextEdit()
        self.sources.setReadOnly(True)
        self.sources.setPlaceholderText("Источники (таймкоды) для последнего ответа…")
        self.sources.setMaximumHeight(220)

        # --- input row ---
        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Задай вопрос по видео…")
        self.btn_send = QPushButton("Отправить")
        self.btn_send.clicked.connect(self.on_send)

        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.btn_send)

        self.status = QLabel("Готово.")
        self.status.setAlignment(Qt.AlignLeft)

        layout.addWidget(box)
        layout.addWidget(self.chat, 1)
        layout.addWidget(QLabel("Источники (для последнего ответа):"))
        layout.addWidget(self.sources)
        layout.addLayout(input_row)
        layout.addWidget(self.status)

        self.load_chat_if_exists()

        transcript_path = self.item_dir / "transcript.json"
        if not transcript_path.exists():
            self.set_busy(True, "Нет транскрипта. Сначала нажми 'Транскрибировать' в главном окне.")
        else:
            self.set_busy(False, "Готово.")

    def append_chat(self, who: str, text: str):
        self.chat.append(f"<b>{who}:</b> {text}")

    def set_busy(self, busy: bool, text: str):
        self.status.setText(text)
        self.btn_send.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.btn_rebuild.setEnabled(not busy)
        self.btn_voice.setEnabled(not busy)
        self.mic_seconds.setEnabled(not busy)
        self.mode_combo.setEnabled(not busy)
        self.quiz_count.setEnabled(not busy)
        self.chk_voice_reply.setEnabled(not busy)

    def mark_rebuild(self):
        self._rebuild_next = True
        self.status.setText("Индекс будет пересобран на следующем запросе.")

    def _thread_is_alive(self) -> bool:
        return self._thread is not None and isValid(self._thread) and self._thread.isRunning()

    def on_send(self):
        question = self.input.text().strip()
        mode = self.mode_combo.currentData()

        if mode != "quiz" and not question:
            return

        if self._thread_is_alive():
            QMessageBox.warning(self, "Подожди", "Уже идёт обработка предыдущего запроса.")
            return

        if question:
            self.append_chat("Вы", question)
            self.history.append({"role": "user", "content": question})
            self.input.clear()
        else:
            question = "Сделай мини-тест по лекции"
            self.append_chat("Вы", question)
            self.history.append({"role": "user", "content": question})

        self.save_chat()

        llm_model = self.model_combo.currentText().strip()
        top_k = int(self.topk.value())
        rebuild = self._rebuild_next
        self._rebuild_next = False

        self.run_rag(
            question=question,
            llm_model=llm_model,
            top_k=top_k,
            rebuild=rebuild,
            assistant_mode=str(mode),
            quiz_questions=int(self.quiz_count.value()),
        )

    def run_rag(
        self,
        question: str,
        llm_model: str,
        top_k: int,
        rebuild: bool,
        assistant_mode: str,
        quiz_questions: int,
    ):
        status_map = {
            "answer": "Думаю…",
            "explain": "Объясняю проще…",
            "quiz": "Генерирую мини-тест…",
        }
        self.set_busy(True, status_map.get(assistant_mode, "Думаю…"))

        self._thread = QThread(self)
        self._worker = RagWorker(
            item_dir=self.item_dir,
            question=question,
            history=self.history,
            llm_model=llm_model,
            top_k=top_k,
            rebuild=rebuild,
            assistant_mode=assistant_mode,
            quiz_questions=quiz_questions,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.on_result)
        self._worker.error.connect(self.on_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def _cleanup_thread(self):
        try:
            if self._worker and isValid(self._worker):
                self._worker.deleteLater()
        except Exception:
            pass
        try:
            if self._thread and isValid(self._thread):
                self._thread.deleteLater()
        except Exception:
            pass

        self._worker = None
        self._thread = None

    def on_result(self, result: RagResult):
        answer = result.answer.strip()

        who = "Тест" if result.mode == "quiz" else "Ассистент"
        self.append_chat(who, answer)

        self.history.append({"role": "assistant", "content": answer})
        self.save_chat()

        self._show_contexts(result.contexts)
        self.set_busy(False, "Готово.")

    def on_voice_ask(self):
        if self._thread_is_alive():
            QMessageBox.warning(self, "Подожди", "Уже идёт обработка предыдущего запроса.")
            return

        llm_model = self.model_combo.currentText().strip()
        top_k = int(self.topk.value())
        rebuild = self._rebuild_next
        self._rebuild_next = False
        assistant_mode = str(self.mode_combo.currentData())

        self.set_busy(True, "Готовлю запись с микрофона…")

        self._thread = QThread(self)
        self._worker = VoiceAskWorker(
            item_dir=self.item_dir,
            history=self.history,
            llm_model=llm_model,
            top_k=top_k,
            rebuild=rebuild,
            assistant_mode=assistant_mode,
            quiz_questions=int(self.quiz_count.value()),
            mic_seconds=int(self.mic_seconds.value()),
            stt_model="small",
            stt_device="cpu",
            stt_lang=None,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.setText)
        self._worker.finished.connect(self.on_voice_result)
        self._worker.error.connect(self.on_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def on_voice_result(self, result: VoiceAskResult):
        question = result.question.strip()
        answer = result.answer.strip()

        self.append_chat("Вы 🎤", question)
        self.history.append({"role": "user", "content": question})

        who = "Тест" if result.mode == "quiz" else "Ассистент"
        self.append_chat(who, answer)
        self.history.append({"role": "assistant", "content": answer})
        self.save_chat()

        self._show_contexts(result.contexts)
        self.set_busy(False, "Готово.")

        if self.chk_voice_reply.isChecked():
            self.start_tts(answer)

    def _show_contexts(self, contexts: list[dict]):
        lines = []
        for i, c in enumerate(contexts, start=1):
            lines.append(
                f"[{i}] score={c['score']:.3f}  time={c['start']:.1f}-{c['end']:.1f}s\n"
                f"{c['text']}\n"
            )
        self.sources.setPlainText("\n".join(lines).strip())

    def start_tts(self, text: str):
        if self._tts_thread is not None and isValid(self._tts_thread) and self._tts_thread.isRunning():
            return

        self._tts_thread = QThread(self)
        self._tts_worker = TtsWorker(text)
        self._tts_worker.moveToThread(self._tts_thread)

        self._tts_thread.started.connect(self._tts_worker.run)
        self._tts_worker.finished.connect(self._tts_thread.quit)
        self._tts_worker.error.connect(self.on_error)
        self._tts_worker.error.connect(self._tts_thread.quit)
        self._tts_thread.finished.connect(self._cleanup_tts_thread)

        self._tts_thread.start()

    def _cleanup_tts_thread(self):
        try:
            if self._tts_worker and isValid(self._tts_worker):
                self._tts_worker.deleteLater()
        except Exception:
            pass
        try:
            if self._tts_thread and isValid(self._tts_thread):
                self._tts_thread.deleteLater()
        except Exception:
            pass

        self._tts_worker = None
        self._tts_thread = None

    def on_error(self, msg: str):
        self.append_chat("Ошибка", msg)
        self.set_busy(False, "Ошибка. См. сообщение в чате.")

    def save_chat(self):
        data = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "llm_model": self.model_combo.currentText().strip(),
            "messages": self.history,
        }
        if not self.chat_path.exists():
            data["created_at"] = datetime.now().isoformat(timespec="seconds")

        self.chat_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status.setText("Чат сохранён.")

    def load_chat_if_exists(self):
        if not self.chat_path.exists():
            return

        data = json.loads(self.chat_path.read_text(encoding="utf-8"))
        msgs = data.get("messages", [])

        self.history = msgs
        self.chat.clear()
        for m in self.history:
            if m["role"] == "user":
                self.append_chat("Вы", m["content"])
            else:
                self.append_chat("Ассистент", m["content"])

        self.sources.clear()
        self.status.setText("Чат загружен из сохранения.")

    def delete_chat(self):
        if self.chat_path.exists():
            self.chat_path.unlink()
        self.history = []
        self.chat.clear()
        self.sources.clear()
        self.status.setText("Чат удалён.")
