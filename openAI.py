import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import ollama  # Подключаем Ollama

# Загружаем переменные окружения
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

if not PINECONE_API_KEY:
    raise ValueError("❌ Ошибка: PINECONE_API_KEY не найден в .env!")

# Инициализация Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "cource"

if index_name not in pc.list_indexes().names():
    print(f"❌ Индекс '{index_name}' не найден. Создаём новый...")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

index = pc.Index(index_name)

# Хранение истории диалога
chat_history = []
last_course_title = None  # Название последнего предложенного курса

def check_courses_exist():
    """Проверяет, есть ли курсы в базе данных."""
    try:
        response = index.describe_index_stats()
        total_vectors = response.get("total_vector_count", 0)
        return total_vectors > 0
    except Exception as e:
        print(f"⚠ Ошибка проверки курсов: {e}")
        return False

def get_embedding(text):
    """Получает эмбеддинг через Ollama."""
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response.get("embedding", None)
    except Exception as e:
        print(f"⚠ Ошибка получения эмбеддинга: {e}")
        return None

def find_course_by_title(title):
    """Ищет курс в Pinecone по названию с безопасным доступом к данным."""
    try:
        vector = get_embedding(title)
        if not vector:
            return "❌ Ошибка при получении эмбеддинга для поиска курса."
        results = index.query(vector=vector, top_k=1, include_metadata=True)
        print("Отправляем запрос к Pinecone с эмбеддингом:", vector)

        matches = results.get("matches", [])
        if matches and len(matches) > 0:
            # Безопасное извлечение деталей
            metadata = matches[0].get("metadata", {})
            return metadata.get("details", "❌ Подробности о курсе не найдены.")
        return "❌ Такой курс не найден в базе."
    except Exception as e:
        print(f"⚠ Ошибка поиска курса: {e}")
        return "❌ Ошибка при поиске курса."


def get_course_info(query):
    """Ищет курсы в Pinecone и формирует связный ответ, используя историю диалога."""
    global chat_history, last_course_title

    try:
        print(f"🔵 Запрос пользователя: {query}")

        # Если пользователь просит подробнее, ищем детали последнего курса
        if "подробнее" in query.lower() or "расскажи больше" in query.lower():
            if last_course_title:
                return find_course_by_title(last_course_title)
            else:
                return "❌ Не могу понять, о каком курсе идет речь. Уточните название."

        # Проверяем наличие курсов в базе
        if not check_courses_exist():
            return "❌ В базе данных пока нет курсов."

        vector = get_embedding(query)
        if not vector:
            return "❌ Ошибка при получении эмбеддинга."

        results = index.query(vector=vector, top_k=13, include_metadata=True)

        # Получаем список совпадений
        matches = results.get("matches", [])
        if not matches:
            return "❌ В базе нет подходящих курсов."

        # Формируем контекст из найденных курсов
        context_parts = []
        for match in matches:
            metadata = match.get("metadata", {})
            title = metadata.get("title")
            description = metadata.get("description")
            if not title or not description:
                print(f"⚠ Пропущен курс, отсутствуют нужные поля: {match}")
                continue
            context_parts.append(f"Название: {title}\nОписание: {description}")

        context = "\n\n".join(context_parts)
        if not context:
            return "❌ Курсы не найдены с достаточной информацией."

        # Сохраняем название первого курса для дальнейших уточнений
        if len(matches) > 0:
            first_metadata = matches[0].get("metadata", {})
            last_course_title = first_metadata.get("title", None)

        # Добавляем сообщение пользователя в историю
        chat_history.append({"role": "user", "content": query})
        # Формируем промпт с историей диалога
        prompt = f"""
        Ты — эксперт по IT-курсам и консультант по обучению. Твоя задача — помогать пользователю находить подходящие курсы.

        🔹 Пользователь задал вопрос: "{query}"
        🔹 Ниже приведены курсы из базы данных:
        {context}
        
        Если пользователь просит рассказать о курсах всех то просто выведи их названия и все.
        
        Пожалуйста, составь ответ в виде списка, где каждый пункт содержит:
        - Название курса
        - Основные ключевые моменты курса (краткий перечень тем или преимуществ)

        Если информации недостаточно или подходящих курсов нет, уточни у пользователя, что именно он ищет.

        🚨 **Правила:**
        1. Отвечай только на основе приведённых курсов.
        2. Не выдумывай информацию.
        3. Отвечай строго на русском языке.
        """
        print(prompt)
        messages = [{"role": "system", "content": "Ты — эксперт по IT-курсам."}]
        messages.extend(chat_history[-10:])
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(model="llama3", messages=messages)

        if not response or "message" not in response or "content" not in response["message"]:
            return "❌ Ошибка: пустой ответ от AI."

        chat_history.append({"role": "assistant", "content": response["message"]["content"]})
        return response["message"]["content"]

    except Exception as e:
        print(f"⚠ Ошибка в get_course_info: {e}")
        return "❌ Ошибка при обработке запроса."
