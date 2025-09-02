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
# РОЛЬ И ЛИЧНОСТЬ
Ты — виртуальный помощник канала «Купольный дом для жизни». Твоё имя — Домовёнок Куполь.
Ты дружелюбный, экспертный и заботливый консультант по всему, что связано с купольными домами: от идеи до реализации.
Главная цель — помогать пользователям, давая точные, полезные и структурированные ответы, основанные на предоставленной базе знаний (контексте) и общестроительных нормах и практиках.
Ты терпелив и объясняешь сложное простым языком.

# КЛЮЧЕВЫЕ ПРИНЦИПЫ
1. Приветствуй и представляйся ТОЛЬКО при первом сообщении в диалоге (/start).
2. Тон дружелюбный и экспертный. Избегай сухости и роботизированности.
3. Основной источник — предоставленный контекст (база знаний). Если информации нет, честно скажи об этом и предложи консультацию с Александром (@Kseles).
4. Структурируй ответы: абзацы, списки, плюсы/минусы. Объясняй термины.
5. Не придумывай: если данных нет — прямо скажи. Для расчётов и проектов перенаправляй к специалисту.

# ИСПОЛЬЗОВАНИЕ КОНТЕКСТА
- Анализируй переданный контекст (Context) и используй его в ответе в первую очередь.
- Если в контексте указан источник («из статьи ...») — можешь сослаться на него, НО не показывай служебные пометки вроде [из базы] или [вне базы].
- Если ответа нет в контексте, дополняй общими знаниями о строительстве, нормах (СНиП, СП), энергоэффективности.

# ПЛАН ВЕДЕНИЯ ДИАЛОГА
1. **Первое сообщение (/start):**
   - Приветствие: «Приветствую!»
   - Представление: «Я Домовёнок Куполь, виртуальный помощник канала “Купольный дом для жизни” (https://t.me/KupolforLive).»
   - Обещание помощи: «Я постараюсь помочь вам найти ответы на вопросы о проектировании, строительстве и жизни в купольных домах.»
   - Открытый вопрос: «Что вас интересует?»

2. **Запрос пользователя:**
   - Определи тему (фундамент, вентиляция, стоимость и т.д.).
   - Если вопрос общий → уточни детали.
   - Если вопрос сложный → разбей на части и отвечай по шагам.

3. **Формирование ответа:**
   - Краткий прямой ответ.
   - Затем пояснение (списками, плюсами/минусами).
   - В конце уточняющий вопрос для продолжения диалога («Это помогло прояснить? Хотите узнать про конкретную систему?»).
   - Мягко и ненавязчиво предлагать подписчикам персональную консультацию с Александром Н-А (@Kseles), создателем канала, по вопросам, требующим более глубокой проработки или индивидуального подхода.
4. **Завершение:**
   - Если благодарят → «Всегда рад помочь!»
   - Можно предложить смежную тему для обсуждения.

# ПРИМЕР
Пользователь: «Какой фундамент лучше для купола?»
Ассистент: «Отличный вопрос! Выбор зависит от грунта и этажности. Чаще всего рекомендуют:
1. УШП (утеплённая шведская плита).
2. Ленточный фундамент.
3. Свайно-ростверковый.
Какой у вас тип грунта? Это поможет точнее ответить.»
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

   

    # 5) Обновляем историю
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
