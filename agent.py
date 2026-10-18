"""
agent.py — агент для анализа исторических личностей на изображениях.

Оркестрирует два шага:
  1. image_search_tool.search_similar_images — ищет похожие изображения в сети
     и собирает контекстную информацию о них.
  2. Multimodal LLM (Mistral через ChatOpenAI) — анализирует изображение
     вместе с найденным веб-контекстом и текстом пользователя.

Точка входа: analyze_image(image_path, user_text) → str
"""

import base64
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from image_search_tool import run_image_search

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Модель ───────────────────────────────────────────────────────────────────

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("Не задан MISTRAL_API_KEY: добавьте его в файл .env (см. .env.example)")

llm = ChatOpenAI(
    openai_api_base="https://api.mistral.ai/v1",
    model="mistral-large-latest",
    api_key=MISTRAL_API_KEY,
    temperature=0,
)

# ─── Системный промпт ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — экспертный исторический ассистент, специализирующийся на распознавании и описании исторических личностей.

## Твоя задача
Проанализировать изображение и предоставить достоверную информацию об изображённой личности, используя:
- прямой визуальный анализ изображения;
- контекстные данные из поиска похожих изображений в интернете (если они предоставлены);
- дополнительный текст от пользователя (если есть).

## Формат ответа

Если личность распознана:
1. Имя — полное имя (+ годы жизни, если известны)
2. Кто это — 1-2 предложения: кем был этот человек и чем прославился
3. Биография — краткое, но содержательное описание (5-8 предложений):
   - происхождение и ранние годы;
   - ключевые достижения или события;
   - историческое значение и наследие.
4. Интересный факт — один нетривиальный факт об этом человеке.
5. Источники — если в веб-контексте были ссылки, упомяни 1-2 наиболее авторитетных.

Если личность не распознана:
- Честно сообщи, что идентифицировать человека не удалось.
- Опиши, что видно на изображении (эпоха, одежда, антураж), если это помогает.
- Не выдумывай имена и биографии.

## Правила
- Опирайся только на достоверные данные. Если не уверен — так и скажи.
- Не смешивай информацию о разных людях.
- Если веб-контекст противоречит визуальному анализу — отметь это явно.
- Отвечай на русском языке, если пользователь не попросил иначе.
- Будь конкретен и информативен, избегай общих фраз.
- ВАЖНО: ВЫВОДИ ТОЛЬКО ЧИСТЫЙ ТЕКСТ. НЕ ИСПОЛЬЗУЙ МАРКДАУН-РАЗМЕТКУ (НИКАКИХ ЗВЕЗДОЧЕК **, РЕШЕТОК # И ПРОЧЕГО).
"""


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _encode_image(image_path: str) -> str:
    """Кодирует изображение в base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_user_message(
    image_base64: str,
    web_context: str,
    user_text: str,
) -> list[dict]:
    """
    Собирает мультимодальный user-блок:
      - системный контекст (результаты веб-поиска + текст пользователя)
      - изображение в base64
    """
    context_parts = []

    if web_context and web_context.strip():
        context_parts.append(
            f"## Данные из веб-поиска похожих изображений\n\n{web_context}"
        )

    if user_text and user_text.strip():
        context_parts.append(f"## Комментарий пользователя\n\n{user_text}")

    context_parts.append(
        "## Задание\n\nПроанализируй изображение с учётом всех данных выше "
        "и дай подробный ответ согласно инструкции."
    )

    text_block = "\n\n".join(context_parts)

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_block},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                },
            ],
        }
    ]


# ─── Основная функция ─────────────────────────────────────────────────────────

def analyze_image(image_path: str, user_text: str = "") -> str:
    """
    Полный пайплайн анализа изображения:

    1. Поиск похожих изображений в сети (image_search_tool).
    2. Формирование обогащённого контекста.
    3. Мультимодальный запрос к LLM с системным промптом.
    4. Возврат итогового ответа.

    Аргументы:
        image_path: путь к локальному файлу изображения.
        user_text:  необязательный текст / подпись от пользователя.

    Возвращает:
        Строку с ответом LLM.
    """
    path = Path(image_path)
    if not path.exists():
        return "Ошибка: изображение не найдено."

    # ── Шаг 1: веб-поиск похожих изображений ────────────────────────────────
    logger.info("Шаг 1: поиск похожих изображений в сети...")
    try:
        web_context = run_image_search(image_path)
        logger.info("Веб-контекст получен (%d символов)", len(web_context))
    except Exception as e:
        logger.warning("Поиск изображений завершился с ошибкой: %s", e)
        web_context = ""

    # ── Шаг 2: кодирование изображения ──────────────────────────────────────
    logger.info("Шаг 2: кодирование изображения в base64...")
    image_base64 = _encode_image(image_path)

    # ── Шаг 3: формирование запроса и вызов LLM ──────────────────────────────
    logger.info("Шаг 3: отправка запроса в LLM...")
    messages = _build_user_message(
        image_base64=image_base64,
        web_context=web_context,
        user_text=user_text,
    )

    # Prepend system prompt as a system message
    full_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,
    ]

    try:
        response = llm.invoke(full_messages)
        result = response.content
    except Exception as e:
        logger.exception("Ошибка при вызове LLM")
        return f"Ошибка при обращении к модели: {e}"

    logger.info("Анализ завершён.")
    return result or "Модель вернула пустой ответ."


def generate_quiz(text_about_person: str, question_count: int = 5) -> str:
    """Генерирует квиз в формате JSON на основе текста о личности."""
    prompt = f"""Ты — интеллектуальный помощник. Твоя задача составить интересный квиз (тест) ровно из {question_count} вопросов по тексту о личности ниже.
Каждый вопрос должен иметь 4 варианта ответов, из которых только один правильный.
Не повторяй вопросы и не делай больше или меньше {question_count} вопросов.
Выведи результат СТРОГО в формате валидного JSON без разметки (без ```json), вот так:
[
  {{
    "question": "Текст вопроса",
    "options": [
      {{"text": "Вариант 1", "is_correct": false}},
      {{"text": "Вариант 2", "is_correct": true}},
      {{"text": "Вариант 3", "is_correct": false}},
      {{"text": "Вариант 4", "is_correct": false}}
    ]
  }},
  ...
]

Текст о личности:
{text_about_person}
"""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = llm.invoke(messages)
        result = response.content.strip()
        # Clean up any potential markdown code blocks
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        return result.strip()
    except Exception as e:
        logger.exception("Ошибка при генерации квиза")
        return "[]"
