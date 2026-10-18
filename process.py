"""
process.py — точка входа для обработки изображения.

Делегирует всю логику агенту (agent.py), который:
  - ищет похожие изображения в сети (image_search_tool);
  - формирует обогащённый контекст;
  - вызывает LLM с обновлённым системным промптом.

Интерфейс функции process_image не изменился,
поэтому main.py не требует правок.
"""

from agent import analyze_image, generate_quiz


def process_image(image_path: str, text: str = "") -> str:
    """
    Анализирует изображение и возвращает текстовый ответ.

    Аргументы:
        image_path: путь к локальному файлу изображения.
        text:       необязательный текст / подпись от пользователя.

    Возвращает:
        Строку с результатом анализа от LLM-агента.
    """
    return analyze_image(image_path=image_path, user_text=text)

def create_quiz_for_person(text: str, question_count: int = 5) -> str:
    """Генерирует квиз по тексту о личности."""
    return generate_quiz(text, question_count)
