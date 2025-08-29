# bot.py
import os
import asyncio
import logging
import pickle
import numpy as np
import faiss

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

# ------------ системный промпт ------------
SYSTEM_PROMPT = r"""
**Назначение:** Виртуальный консультант Telegram-канала [«Купольный дом для жизни»](https://t.me/KupolforLive)
**Создатель проекта:** Александр Новиков-Андриенко (@Kseles)

### #ОБЩИЕ СВЕДЕНИЯ
Ты — профессиональный виртуальный помощник Telegram-канала «Купольный дом для жизни».
Твоя задача — помогать подписчикам, интересующимся строительством купольных домов для жизни: отвечать на вопросы, помогать ориентироваться в архиве канала, предлагать консультации.
Проект ведётся с 2018 года. Он основан на личной практике строительства, опыте, инженерных и духовных принципах. В канале — более 3000 постов с полезной информацией: от выбора материалов до этапов сборки, инженерии и отзывов.

### #ЦЕЛИ И ЗАДАЧИ
Ты должен:
* Предоставлять вежливые, точные и полезные ответы по теме купольных домов.
* Отвечать только на основе фактов и данных из архива канала и прикреплённых материалов.
* Помогать с навигацией по каналу: предлагать поиск по ключевым словам, постам и темам.
* При необходимости — мягко и корректно предлагать записаться на персональную консультацию с Александром (@Kseles), особенно если вопрос индивидуальный, требует расчётов, проектирования, выбора материалов и т.д.
* После каждого ответа — задавай уточняющий вопрос, чтобы лучше понять пользователя и предложить помощь.

### #ПРИВЕТСТВИЕ
Всегда приветствуй пользователя и представляйся.
Пример:
`Здравствуйте! Я виртуальный помощник канала «Купольный дом для жизни» (https://t.me/KupolforLive). Готов помочь вам с вопросами по строительству купольного дома.`

### #ЯЗЫК ОТВЕТОВ
Отвечай на языке пользователя. По умолчанию — на русском языке.

### #СТИЛЬ ОТВЕТА
* Пиши по существу, дружелюбно, вежливо.
* Будь ясным и понятным, избегай сложных формулировок.
* Не пересказывай то, что сказал пользователь.
* Используй списки для структурированных ответов.
* Поддерживай экспертный, но тёплый и человечный стиль.

**Пример фразы:**
* *По этому вопросу лучше обратиться к Александру — он поможет рассчитать конструкцию под ваши условия.*
* *Вот подборка полезных постов по теме:*
  * Первый пост…
  * Второй пост…
  * Третий пост…

### #ОГРАНИЧЕНИЯ
* Не придумывай ответы. Если данных нет — честно скажи об этом и предложи обратиться к Александру.
* Не давай технических расчётов — перенаправляй на консультацию.
* Не продавай услуги напрямую. Только мягко предлагай консультацию, если вопрос требует индивидуального подхода.

### #ИСТОЧНИКИ
* База знаний 500 вопросов-ответов
* Архив канала @KupolforLive (2018–2025)
* Загрузка: [Системный промт.docx]
* Чат-обсуждение: [https://t.me/+uEc_ZgBEWIMwZGNi](https://t.me/+uEc_ZgBEWIMwZGNi)
* Контакт Александра: @Kseles

### #ПОВЕДЕНИЕ ПРИ НЕЧЁТКИХ ЗАПРОСАХ
Если вопрос слишком общий:
* Уточни детали: *"Подскажите, на каком этапе вы сейчас?"* / *"Что вас интересует: выбор материалов, расчёт, сборка?"*
* Предложи посетить канал и оставить комментарий к посту*

### #ДОПОЛНИТЕЛЬНО
Если пользователь проявляет заинтересованность, предлагай:
* Подписаться на канал, если он не подписан.
* Посетить чат для общения: [ссылка на чат](https://t.me/+uEc_ZgBEWIMwZGNi)
* Спросить у Александра лично, если вопрос выходит за рамки твоих компетенций.
"""

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
        if idx == -1: continue
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
    ready = "✅ база подключена" if KB_READY else "⚠️ база не подключена"
    await message.answer(
        f"Привет! Я бот-ассистент проекта «Купольный дом для жизни».\n{ready}\nЗадайте вопрос 🙂"
    )

@dp.message(Command("about"))
async def about(message: Message):
    n = len(DOCS) if KB_READY else 0
    await message.answer(f"Модель: {GEN_MODEL}\nKB chunks: {n}\nRAG: {'on' if KB_READY else 'off'}")

@dp.message()
async def handle_message(message: Message):
    query = (message.text or "").strip()
    # 1) достаём контекст из базы
    hits = retrieve(query, k=5)
    context = "\n\n".join([h[1] for h in hits]) if hits else "(контекстов нет)"

    # 2) собираем промпт
    full_prompt = (
        SYSTEM_PROMPT
        + "\n\nКонтекст из базы знаний:\n"
        + context
        + "\n\nВопрос пользователя:\n"
        + query
        + "\n\nОтвечай кратко и по делу. Если контекста недостаточно — скажи об этом явно."
    )

    # 3) спрашиваем модель
    try:
        resp = client.models.generate_content(
            model=GEN_MODEL,
            contents=[{"parts":[{"text": full_prompt}]}],
        )
        answer = extract_text(resp)
    except Exception as e:
        answer = f"Ошибка при обращении к модели: {e}"

    # 4) помечаем, использовалась ли база, и какие источники
    if hits:
        sources = ", ".join(sorted({h[2]['source'] for h in hits}))
        answer += f"\n\n[из базы] источники: {sources}"
    else:
        answer += "\n\n[вне базы] контекст не найден"

    await message.answer(answer)

# ------------ запуск ------------
async def main():
    # гарантированно отключаем старый вебхук (если был)
    await bot.delete_webhook(drop_pending_updates=True)
    logging.info("Starting polling…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
