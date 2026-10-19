from pathlib import Path
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from scipy.io.wavfile import write
import numpy as np

# Моделі Hugging Face
VISION_MODEL = "Salesforce/blip-image-captioning-base"
VOICE_MODEL = "facebook/mms-tts-eng"

# Папки проєкту
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Моделі завантажуються лише один раз
vision_ai = None
voice_ai = None

# Останній отриманий опис
last_description = ""
last_image = ""

# Завантаження моделей
def load_vision():
    global vision_ai
    if vision_ai is None:
        print("\nЗавантаження моделі розпізнавання зображень...")
        vision_ai = pipeline(
            "image-to-text",
            model=VISION_MODEL,
            device=-1
        )
    return vision_ai


# Завантаження голосової моделі
def load_voice():
    global voice_ai
    if voice_ai is None:
        print("\nЗавантаження голосової моделі...")
        voice_ai = pipeline(
            "text-to-speech",
            model=VOICE_MODEL,
            device=-1
        )
        print("Голосова модель завантажена!")
    return voice_ai


# Вибір фотографії
def choose_image():
    print("\nДоступні фотографії:")
    images = [
        file for file in sorted(INPUT_DIR.iterdir())
        if file.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    if not images:
        print("У папці input немає фотографій.")
        return None
    for i, image in enumerate(images, 1):
        print(f"{i}. {image.name}")
    choice = input("\nВведіть номер або шлях до фото: ").strip().strip('"')
    if choice.isdigit():
        number = int(choice)
        if 1 <= number <= len(images):
            return images[number - 1]
        print("Невірний номер фотографії!")
        return None
    # Можна ввести абсолютний або відносний шлях
    path = Path(choice)
    if not path.is_absolute():
        if path.is_file():
            path = path.resolve()
        else:
            path = BASE_DIR / choice
    if not path.is_file():
        print("Файл не знайдено!")
        return None
    if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        print("Підтримуються лише JPG, JPEG та PNG.")
        return None
    return path


# Створення опису фотографії
def describe_image(image_path):
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        print("Помилка відкриття зображення!")
        return None
    print("\nАналіз фотографії...")
    model = load_vision()
    result = model(image, max_new_tokens=60)
    if not result or not result[0].get("generated_text"):
        print("Модель не змогла створити опис.")
        return None
    description = result[0]["generated_text"].strip()
    if not description:
        print("Отримано порожній опис.")
        return None
    print("\nОпис зображення:")
    print(description)
    return description


# Збереження текстового опису
def save_description(image_name, description):
    path = OUTPUT_DIR / "descriptions.txt"
    with open(path, "a", encoding="utf-8") as file:
        file.write(f"Фото: {image_name}\n")
        file.write(f"Опис: {description}\n")
        file.write("-" * 45 + "\n")


# Вибір назви аудіофайлу
def get_audio_name(default_name):
    name = input(
        f"\nНазва аудіофайлу (Enter - {default_name}): "
    ).strip()
    if not name:
        name = default_name
    # Прибираємо розширення, якщо користувач його ввів
    name = Path(name).stem
    # Залишаємо лише безпечні символи
    name = "".join(
        char for char in name
        if char.isalnum() or char in "-_"
    )
    if not name:
        name = "description"
    path = OUTPUT_DIR / f"{name}.wav"
    # Не перезаписуємо наявний файл
    counter = 1
    while path.exists():
        path = OUTPUT_DIR / f"{name}_{counter}.wav"
        counter += 1
    return path


# Перетворення тексту на голос
def generate_speech(description, default_name):
    print("\nГенерація голосового опису...")
    model = load_voice()
    audio_result = model(description)
    audio = np.asarray(audio_result["audio"], dtype=np.float32)
    audio = audio.squeeze()
    if audio.size == 0:
        raise ValueError("Модель повернула порожнє аудіо.")
    sampling_rate = int(audio_result["sampling_rate"])
    audio_path = get_audio_name(default_name)
    write(str(audio_path), sampling_rate, audio)
    print("\nАудіо успішно створено!")
    print("Збережено:", audio_path)
    return audio_path


# Обробка однієї фотографії
def process_image(image_path):
    global last_description, last_image
    description = describe_image(image_path)
    if description is None:
        return False
    save_description(image_path.name, description)
    # Зберігаємо опис для повторного озвучення
    last_description = description
    last_image = image_path.name
    audio_path = generate_speech(description, image_path.stem)
    print("\n     RESULT")
    print("Фото:", image_path.name)
    print("Опис:", description)
    print("Модель зображень:", VISION_MODEL)
    print("Голосова модель:", VOICE_MODEL)
    print("Аудіофайл:", audio_path)
    print("Текстові описи:", OUTPUT_DIR / "descriptions.txt")
    return True


# Аналіз однієї фотографії
def analyze_one():
    image_path = choose_image()
    if image_path is not None:
        process_image(image_path)


# Аналіз усіх фотографій
def analyze_all():
    images = [
        file for file in sorted(INPUT_DIR.iterdir())
        if file.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    if not images:
        print("Немає фотографій для аналізу!")
        return
    print(f"\nЗнайдено фотографій: {len(images)}")
    completed = 0
    for image_path in images:
        print("\n" + "=" * 45)
        print("Фото:", image_path.name)
        try:
            if process_image(image_path):
                completed += 1
        except Exception as error:
            print("Помилка обробки:", error)

    print("\n     REPORT")
    print("Усього фотографій:", len(images))
    print("Завершено без помилок:", completed)
    print("Помилок:", len(images) - completed)


# Повторне озвучення
def repeat_speech():
    if not last_description:
        print("\nСпочатку потрібно проаналізувати фотографію!")
        return
    print("\nОстанній опис:")
    print(last_description)
    generate_speech(last_description, "repeat_description")


# Головне меню
def show_menu():
    print("\n      AI VISION & VOICE ASSISTANT")
    print("1. Аналіз однієї фотографії")
    print("2. Аналіз усіх фотографій")
    print("3. Повторне озвучення опису")
    print("4. Показати використані моделі")
    print("0. Вихід")


def show_models():
    print("\nМодель для аналізу зображень:")
    print(VISION_MODEL)
    print("\nМодель для синтезу голосу:")
    print(VOICE_MODEL)
    print("\nМова опису та озвучення: англійська")


# Запуск програми
def main():
    while True:
        show_menu()
        choice = input("\nОберіть дію: ")
        try:
            if choice == "1":
                analyze_one()
            elif choice == "2":
                analyze_all()
            elif choice == "3":
                repeat_speech()
            elif choice == "4":
                show_models()
            elif choice == "0":
                print("\nДо побачення!")
                break
            else:
                print("Невірний вибір!")
        except Exception as error:
            print("\nПомилка:", error)
            print("Перевірте модель, бібліотеки та файли.")


if __name__ == "__main__":
    main()
