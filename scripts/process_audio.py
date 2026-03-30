import os
import sys
import asyncio
import argparse
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения из текущей директории или из корня бота
load_dotenv()
load_dotenv(os.path.join(os.getcwd(), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(os.path.join("C:\\CODE\\nanobot", ".env"))

# Инициализируем клиента
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Ошибка: OPENAI_API_KEY не найден в переменных окружения.")
    sys.exit(1)

client = AsyncOpenAI(api_key=api_key)

# Путь к папке отчетов относительно самого скрипта.
# Если скрипт лежит в workspace/scripts/process_audio.py,
# то отчеты будут сохраняться in workspace/reports/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "reports"))

async def transcribe(audio_path):
    print(f"Начинаю транскрибацию файла: {audio_path}...")
    try:
        with open(audio_path, "rb") as audio:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language="ru"
            )
        return response.text
    except Exception as e:
        print(f"Ошибка при транскрибации: {e}")
        return None

async def summarize(transcript):
    print("Генерирую последовательный отчет на русском языке...")
    system_prompt = (
        "Ты — профессиональный ассистент по подготовке отчетов.\n"
        "Твоя задача — составить последовательный отчет по предоставленной транскрипции аудио на русском языке.\n\n"
        "Правила формирования отчета:\n"
        "1. **Последовательность**: Описывай события и темы в том порядке, в котором они звучат в записи.\n"
        "2. **Приоритетность**: Если какая-то тема обсуждается дольше или подробнее (большой объем текста в транскрипции), "
        "удели ей больше внимания в отчете, распиши детали.\n"
        "3. **Структура**: Используй нумерованные списки или заголовки для разделения этапов.\n"
        "4. **Стиль**: Деловой, четкий, без лишней воды, но с сохранением всех важных деталей.\n"
        "5. **Язык**: Русский.\n"
        "6. **Формат**: Markdown."
    )
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Вот текст транскрипции:\n\n{transcript}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ошибка при суммаризации: {e}")
        return None

async def main():
    parser = argparse.ArgumentParser(description="Транскрибация и суммаризация аудио для Nanobot.")
    parser.add_argument("audio_path", help="Путь к аудиофайлу")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Ошибка: Файл {args.audio_path} не найден!")
        return

    transcript = await transcribe(args.audio_path)
    if not transcript:
        return

    report = await summarize(transcript)
    if not report:
        return

    # Создаем директорию отчетов, если её нет
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Сохраняем отчет с таймстампом
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"{timestamp}_report.md"
    report_path = os.path.join(REPORTS_DIR, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nВсе готово!")
    print(f"Отчет сохранен в: {report_path}")
    print("-" * 20)
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
