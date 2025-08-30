# bot.py
import os
import asyncio
import logging
import pickle
import numpy as np
import faiss
from collections import defaultdict, deque

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from dotenv import load_dotenv
from google import genai

# ------------ базовые настройки ------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY   = os.getenv("GEMINI_API_KEY")
if not BOT_TOKEN:
    raise ValueError("❌ Нет BOT_TOKEN в .env")
if not API_KEY:
    raise ValueError("❌ Нет GEMINI_API_KEY в .env")

bot = Bot(BOT_TOKEN)
dp  = Dispatcher()
client = genai.Client(api_key=API_KEY)

# ------------ системный промпт (обновлён) ------------
SYSTEM_PROMPT = r"""
РОЛЬ: виртуальный помощник Telegram-канала «Купольный дом для жизни».

ОСНОВАНИЕ ДЛЯ ОТВЕТОВ
- Опирайся на контекст из базы знаний и историю диалога ниже.
- Если данных нет — прямо скажи об этом и предложи консультацию у Александра (@Kseles).

СТИЛЬ
- Кратко, по делу, дружелюбно. Без канцелярита и повтора вопроса.
- Списки — когда уместно. В конце — 1 уточняющий вопрос.

ПРАВИЛА ДИАЛОГА
- НЕ представляйся повторно. Приветствие только в /start или по просьбе пользователя.
- Помни контекст последних сообщений: если пользователь отвечает «да/ок/ага/верно/подходит» — считай это согласием на твоё предыдущее предложение и продолжай без переспроса.
- Если запрос общий — уточни максимум 1–2 детали.
- Если нужен расчёт/проектирование/индивидуальные условия — мягко предложи консультацию с @Kseles.

ФОРМАТ
- Короткий ответ → при необходимости пункты/шаги → 1 уточняющий вопрос.
- Если использована база — можно добавить «могу прислать источники» (не обязательно каждый раз).

ОГРАНИЧЕНИЯ
- Не выдумывай. Нет данных — так и скажи.
- Без технических расчётов — отправляй на консультацию.
"""

# ------------ локальная память диалога ------------
YES_WORDS = {"да", "ок", "ага", "конечно", "подходит", "верно", "согласен", "согласна"}
user_state = defaultdict(lambda: {
    "greeted": False,
    "history": deque(maxlen=6),   # [("user", txt), ("assistant", txt), ...]
})

def make_history_text(history):
    if not history:
        return "(пусто)"
    lines = []
    for role, txt in history:
        prefix = "Пользователь:" if role == "user" else "Ассистент:"
        lines.append(f"{prefix} {txt}")
    return "\n".join(lines)

# ------------ RAG: загрузка индекса ------------
EMBED_MODEL = "text-embedding-004"
GEN_MODEL   = "gemini-2.0-flash"

try:
    INDEX = faiss.read_index("kb.index")
    KB = pickle.load(open("kb.pkl", "rb"))
    DOCS, META = KB["docs"], KB["meta"]
    logging.info("KB loaded: %d chunks", len(DOCS))
    KB_READY = True
except Exception as e:
    logging.warning("KB not loaded (%s). Bot will work без базы.", e)
    DOCS, META, INDEX = [], [], None
    KB_READY = False

def _embed(text: str) -> np.ndarray:
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[{"parts":[{"text": text}]}]
    )
    v = np.array(resp.embeddings[0].values, dtype="float32")
    v /= (np.linalg.norm(v) + 1e-12)
    return v.reshape(1, -1)

def retrieve(query: str, k: int = 5):
    if not KB_READY or INDEX is None or not DOCS:
        return []
    qv = _embed(query)
    D, I = INDEX.search(qv, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        hits.append((float(score), DOCS[idx], META[idx]))
    return hits

def extract_text(resp) -> str:
    try:
        parts = resp.candidates[0].content.parts
        texts = []
        for p in parts:
            if hasattr(p, "text") and p.text:
                texts.append(p.text)
        return "\n".join(texts).strip() or "❌ Модель не вернула текста."
    except Exception:
        return "❌ Модель не вернула текста."

# ------------ хэндлеры ------------
@dp.message(Command("start"))
async def start(message: Message):
    st = user_state[message.from_user.id]
    st["greeted"] = True
    ready = "✅ база подключена" if KB_READY else "⚠️ база не подключена"
    await message.answer(f"Привет! Я бот-ассистент проекта «Купольный дом для жизни».\n{ready}\nЗадайте вопрос 🙂")

@dp.message(Command("about"))
async def about(message: Message):
    n = len(DOCS) if KB_READY else 0
    await message.answer(f"Модель: {GEN_MODEL}\nKB chunks: {n}\nRAG: {'on' if KB_READY else 'off'}")

@dp.message()
async def handle_message(message: Message):
    uid = message.from_user.id
    st = user_state[uid]

    query_raw = (message.text or "").strip()
    is_short_yes = query_raw.lower() in YES_WORDS

    # 1) История диалога и формулировка пользовательского входа
    history_text = make_history_text(st["history"])
    if is_short_yes:
        user_turn = "Пользователь подтвердил согласие на твоё предыдущее предложение. Продолжи следующий логичный шаг без повторного представления."
        search_query = "продолжение предыдущего предложения по теме разговора"
    else:
        user_turn = f"Новый вопрос пользователя: {query_raw}"
        search_query = query_raw

    # 2) RAG: контексты
    hits = retrieve(search_query, k=5)
    context = "\n\n".join([h[1] for h in hits]) if hits else "(контекстов нет)"

    # 3) Полный промпт
    full_prompt = (
        SYSTEM_PROMPT
        + "\n\nИстория диалога (последние реплики):\n"
        + history_text
        + "\n\nКонтекст из базы знаний:\n"
        + context
        + "\n\n"
        + user_turn
        + "\n\nСформируй ответ согласно правилам."
    )

    # 4) Вызов модели
    try:
        resp = client.models.generate_content(
            model=GEN_MODEL,
            contents=[{"parts":[{"text": full_prompt}]}],
        )
        answer = extract_text(resp)
    except Exception as e:
        answer = f"Ошибка при обращении к модели: {e}"

    # 5) Пометка источников
    if hits:
        sources = ", ".join(sorted({h[2]['source'] for h in hits}))
        answer += f"\n\n[из базы] источники: {sources}"
    else:
        answer += "\n\n[вне базы] контекст не найден"

    # 6) Обновляем историю
    if query_raw:
        st["history"].append(("user", query_raw))
    st["history"].append(("assistant", answer))

    await message.answer(answer)

# ------------ запуск ------------
async def main():
    await bot.delete_webhook(drop_pending_updates=True)  # на всякий случай
    logging.info("Starting polling…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
