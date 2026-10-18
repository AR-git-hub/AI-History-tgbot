import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN: добавьте его в файл .env (см. .env.example)")

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler,
    filters,
)

import json
from process import process_image, create_quiz_for_person


WAITING_IMAGE = 1
LOW_SCORE_THRESHOLD = 40
QUIZ_QUESTION_COUNTS = (5, 10, 15)
DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)
QUIZ_HISTORY_FILE = Path("quiz_history.json")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def _load_quiz_history() -> dict:
    if not QUIZ_HISTORY_FILE.exists():
        return {"users": {}}

    try:
        with QUIZ_HISTORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logging.exception("Не удалось прочитать историю квизов")
        return {"users": {}}

    if not isinstance(data, dict):
        return {"users": {}}
    data.setdefault("users", {})
    return data


def _save_quiz_history(data: dict) -> None:
    with QUIZ_HISTORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _extract_quiz_title(person_text: str) -> str:
    for line in person_text.splitlines():
        line = line.strip()
        if not line:
            continue

        lowered = line.lower()
        for prefix in ("имя —", "имя -", "имя:"):
            if lowered.startswith(prefix):
                title = line.split(prefix[-1], 1)[-1].strip()
                return title[:80] or "Без названия"

        return line[:80]

    return "Без названия"


def _get_user_quiz_data(history: dict, user_id: int) -> dict:
    users = history.setdefault("users", {})
    user_data = users.setdefault(str(user_id), {})
    user_data.setdefault("attempts", [])
    user_data.setdefault("weak_topics", [])
    return user_data


def _record_quiz_attempt(
    user_id: int,
    person_text: str,
    correct: int,
    total: int,
) -> tuple[int, bool]:
    percent = round(correct / total * 100) if total else 0
    title = _extract_quiz_title(person_text)
    now = datetime.now().isoformat(timespec="seconds")

    history = _load_quiz_history()
    user_data = _get_user_quiz_data(history, user_id)
    user_data["attempts"].append(
        {
            "title": title,
            "person_text": person_text,
            "correct": correct,
            "total": total,
            "percent": percent,
            "created_at": now,
        }
    )

    is_low_score = percent < LOW_SCORE_THRESHOLD
    if is_low_score:
        weak_topics = user_data["weak_topics"]
        existing = next(
            (topic for topic in weak_topics if topic.get("title") == title),
            None,
        )
        topic_data = {
            "title": title,
            "person_text": person_text,
            "last_percent": percent,
            "last_result": f"{correct}/{total}",
            "updated_at": now,
        }
        if existing:
            existing.update(topic_data)
        else:
            weak_topics.append(topic_data)

    _save_quiz_history(history)
    return percent, is_low_score


def _get_weak_topics(user_id: int) -> list[dict]:
    history = _load_quiz_history()
    user_data = _get_user_quiz_data(history, user_id)
    return user_data["weak_topics"]


def _build_quiz_size_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(
                f"{question_count} вопросов",
                callback_data=f"quiz_count_{question_count}",
            )
        ]
        for question_count in QUIZ_QUESTION_COUNTS
    ]
    return InlineKeyboardMarkup(keyboard)


def _trim_for_telegram(text: str, limit: int = 3900) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\nОтветы не поместились целиком из-за лимита Telegram."


async def _ask_quiz_question_count(message, context: ContextTypes.DEFAULT_TYPE, person_text: str):
    context.user_data["pending_quiz_person_text"] = person_text
    await message.reply_text(
        "Сколько вопросов сделать в квизе?",
        reply_markup=_build_quiz_size_keyboard(),
    )


async def _edit_quiz_question_count(message, context: ContextTypes.DEFAULT_TYPE, person_text: str):
    context.user_data["pending_quiz_person_text"] = person_text
    await message.edit_text(
        "Сколько вопросов сделать в квизе?",
        reply_markup=_build_quiz_size_keyboard(),
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Доступные команды:\n"
        "/help — описание\n"
        "/find — отправить изображение на обработку\n"
        "/quiz — пройти тест по последней распознанной личности\n"
        "/history — темы, где результат был ниже 40%\n"
        "/cancel — отмена"
    )

async def start_quiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'last_person_text' not in context.user_data:
        await update.message.reply_text("Сначала отправьте изображение через /find, чтобы я распознал личность!")
        return

    await _ask_quiz_question_count(update.message, context, context.user_data["last_person_text"])

async def handle_generate_quiz_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if 'last_person_text' not in context.user_data:
        await query.edit_message_text("Данные устарели. Пожалуйста, отправьте изображение заново через /find.")
        return

    await _edit_quiz_question_count(query.message, context, context.user_data["last_person_text"])

async def handle_quiz_count_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    try:
        question_count = int(query.data.rsplit("_", 1)[-1])
    except ValueError:
        await query.edit_message_text("Не удалось выбрать количество вопросов. Попробуйте еще раз.")
        return

    if question_count not in QUIZ_QUESTION_COUNTS:
        await query.edit_message_text("Можно выбрать только 5, 10 или 15 вопросов.")
        return

    person_text = context.user_data.get("pending_quiz_person_text")
    if not person_text:
        await query.edit_message_text("Данные квиза устарели. Запустите /quiz или /history еще раз.")
        return

    await query.edit_message_text(
        f"Генерирую квиз на {question_count} вопросов... Это может занять несколько секунд ⏳"
    )
    await _start_dynamic_quiz(query.message, context, person_text, question_count)

async def handle_retry_weak_quiz_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    try:
        topic_index = int(query.data.rsplit("_", 1)[-1])
    except ValueError:
        await query.edit_message_text("Не удалось выбрать тему. Откройте /history еще раз.")
        return

    weak_topics = _get_weak_topics(query.from_user.id)
    if topic_index < 0 or topic_index >= len(weak_topics):
        await query.edit_message_text("Эта тема больше не найдена. Откройте /history еще раз.")
        return

    topic = weak_topics[topic_index]
    await _edit_quiz_question_count(query.message, context, topic["person_text"])

async def _start_dynamic_quiz(
    message,
    context: ContextTypes.DEFAULT_TYPE,
    person_text: str | None = None,
    question_count: int = 5,
):
    person_text = person_text or context.user_data['last_person_text']
    try:
        quiz_json_str = create_quiz_for_person(person_text, question_count)
        quiz_data = json.loads(quiz_json_str)
        if not isinstance(quiz_data, list) or len(quiz_data) == 0:
            raise ValueError("Неверный формат квиза")
    except Exception as e:
        logging.error(f"Error parsing quiz: {e}")
        await message.edit_text("Не удалось сгенерировать квиз. Попробуйте еще раз позже.")
        return
        
    context.user_data['current_quiz_questions'] = quiz_data
    context.user_data['current_quiz_person_text'] = person_text
    context.user_data['quiz_correct'] = 0
    context.user_data['quiz_results'] = []
    await send_dynamic_question(message, context, 0)

async def send_dynamic_question(message, context: ContextTypes.DEFAULT_TYPE, q_index: int):
    quiz_data = context.user_data['current_quiz_questions']
    q_data = quiz_data[q_index]
    
    keyboard = []
    for opt in q_data['options']:
        is_correct = "1" if opt.get('is_correct') else "0"
        keyboard.append([InlineKeyboardButton(opt['text'], callback_data=f"dq_{q_index}_{is_correct}")])
        
    reply_markup = InlineKeyboardMarkup(keyboard)
    text = f"Вопрос {q_index + 1} из {len(quiz_data)}\n\n{q_data['question']}"
    
    await message.edit_text(text, reply_markup=reply_markup)

async def handle_dynamic_quiz_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    parts = data.split("_")
    q_index = int(parts[1])
    is_correct = int(parts[2])
    
    quiz_data = context.user_data.get('current_quiz_questions', [])
    
    if is_correct:
        context.user_data['quiz_correct'] += 1
        context.user_data['quiz_results'].append(True)
    else:
        context.user_data['quiz_results'].append(False)
        
    next_q_index = q_index + 1
    if next_q_index < len(quiz_data):
        await send_dynamic_question(query.message, context, next_q_index)
    else:
        correct = context.user_data.get('quiz_correct', 0)
        total = len(quiz_data)
        person_text = context.user_data.get('current_quiz_person_text', "")
        percent, is_low_score = _record_quiz_attempt(
            user_id=query.from_user.id,
            person_text=person_text,
            correct=correct,
            total=total,
        )
        
        summary = f"🎯 Квиз завершен!\n\nВы ответили правильно на {correct} из {total} вопросов ({percent}%).\n"
        if is_low_score:
            summary += "Результат ниже 40%, поэтому я сохранил эту тему в /history для повторения с новыми вопросами.\n"
        summary += "\n📚 Сводка правильных ответов:\n"
        results = context.user_data.get('quiz_results', [])
        for i, q in enumerate(quiz_data, 1):
            correct_opt = next((opt['text'] for opt in q['options'] if opt.get('is_correct')), "Нет данных")
            user_was_correct = results[i-1] if i-1 < len(results) else False
            emoji = "✅" if user_was_correct else "❌"
            summary += f"\n{i}. {q['question']}\n{emoji} Правильный: {correct_opt}"
        
        await query.edit_message_text(_trim_for_telegram(summary))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "После команды /find отправь картинку.\n"
        "Можно также добавить подпись к изображению.\n"
        "/history покажет темы, где результат квиза был ниже 40%."
    )

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    weak_topics = _get_weak_topics(update.effective_user.id)
    if not weak_topics:
        await update.message.reply_text(
            "Пока нет тем для повторения. Они появятся здесь, если пройти квиз меньше чем на 40%."
        )
        return

    keyboard = []
    lines = ["Темы для повторения:"]
    for index, topic in enumerate(weak_topics):
        title = topic.get("title", "Без названия")
        last_result = topic.get("last_result", "?")
        last_percent = topic.get("last_percent", "?")
        lines.append(f"{index + 1}. {title} — последний результат {last_result} ({last_percent}%)")
        keyboard.append(
            [
                InlineKeyboardButton(
                    f"Повторить: {title[:45]}",
                    callback_data=f"retry_weak_quiz_{index}",
                )
            ]
        )

    await update.message.reply_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(keyboard),
    )

async def find_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Отправь изображение для обработки.\n"
        "Можно с подписью.\n"
        "Для отмены введи /cancel"
    )
    return WAITING_IMAGE

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message

    if message is None or not message.photo:
        await update.message.reply_text("Не вижу изображения. Отправь фото еще раз.")
        return WAITING_IMAGE

    # Берем самое качественное фото из списка размеров
    photo = message.photo[-1]
    caption = (message.caption or "").strip()

    # Получаем файл и скачиваем локально
    telegram_file = await photo.get_file()
    file_path = DOWNLOAD_DIR / f"{message.from_user.id}_{message.message_id}.jpg"
    await telegram_file.download_to_drive(custom_path=str(file_path))

    try:
        # process_image должен вернуть строку
        result_text = process_image(
            image_path=str(file_path),
            text=caption,
        )

        if not result_text:
            result_text = "Обработка завершена, но результат пустой."

        context.user_data['last_person_text'] = result_text
        
        keyboard = [[InlineKeyboardButton("🎯 Пройти квиз по этой личности", callback_data="generate_quiz_btn")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(result_text, reply_markup=reply_markup)

    except Exception as e:
        logging.exception("Ошибка обработки изображения")
        await update.message.reply_text(f"Ошибка при обработке изображения: {e}")

    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Действие отменено.")
    return ConversationHandler.END

async def fallback_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Используй /find, чтобы отправить изображение на обработку."
    )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.exception("Ошибка во время работы бота:", exc_info=context.error)

async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("help", "выводит справку о боте"),
        BotCommand("find", "отправить изображение на обработку"),
        BotCommand("quiz", "пройти тест по последней распознанной личности"),
        BotCommand("history", "повторить темы с результатом ниже 40%"),
        BotCommand("cancel", "отмена отправки"),
    ])

def main():
    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("find", find_command)],
        states={
            WAITING_IMAGE: [
                MessageHandler(filters.PHOTO, handle_image),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("quiz", start_quiz_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CallbackQueryHandler(handle_generate_quiz_btn, pattern="^generate_quiz_btn$"))
    app.add_handler(CallbackQueryHandler(handle_quiz_count_btn, pattern="^quiz_count_"))
    app.add_handler(CallbackQueryHandler(handle_retry_weak_quiz_btn, pattern="^retry_weak_quiz_"))
    app.add_handler(CallbackQueryHandler(handle_dynamic_quiz_answer, pattern="^dq_"))
    app.add_handler(conv_handler)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, fallback_message))
    app.add_error_handler(error_handler)

    app.run_polling()

if __name__ == "__main__":
    main()
