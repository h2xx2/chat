import requests
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Загружаем переменные окружения
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# Проверяем, загружены ли API-ключи
if not DEEPSEEK_API_KEY:
    raise ValueError("❌ Ошибка: переменная DEEPSEEK_API_KEY не найдена в .env!")
if not PINECONE_API_KEY:
    raise ValueError("❌ Ошибка: переменная PINECONE_API_KEY не найдена в .env!")

# Инициализация Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
print("🔵 Доступные индексы в Pinecone:", pc.list_indexes().names())

index_name = "course"

# Если индекс не существует, создаем его
if index_name not in pc.list_indexes().names():
    print(f"❌ Индекс '{index_name}' не найден. Создаём новый...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

index = pc.Index(index_name)

# Создаем клиента OpenAI с вашим DeepSeek API ключом
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def get_embedding(text):
    """Получает эмбеддинг через DeepSeek."""
    try:
        print(f"🔵 Отправка запроса на эмбеддинг для текста: {text}")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": text}
            ],
            stream=False
        )
        # Проверяем, если результат содержит эмбеддинг
        if 'choices' in response:
            embedding = response['choices'][0]['message']['content']
            print(f"🔵 Получен эмбеддинг: {embedding}")
            return embedding
        else:
            raise ValueError(f"❌ Ошибка получения эмбеддинга: {response}")
    except Exception as e:
        print(f"⚠ Ошибка при получении эмбеддинга: {e}")
        return None

def get_course_info(query):
    """Ищет похожие курсы в Pinecone и передает их в DeepSeek."""
    try:
        print(f"🔵 Вызов функции get_course_info с запросом: {query}")

        # Получение эмбеддинга для запроса
        vector = get_embedding(query)

        if vector is None:
            return "❌ Не удалось получить эмбеддинг."

        # Запрос к Pinecone для поиска похожих курсов
        results = index.query(vector, top_k=3, include_metadata=True)

        context = "\n\n".join([  # Формируем контекст для ответа
            f"Название: {match['id']}\nОписание: {match['metadata']['description']}"
            for match in results.get("matches", [])
        ])

        # Формируем запрос для DeepSeek
        prompt = f"""
        Ты — консультант по обучению. Пользователь спрашивает: "{query}". 
        На основе следующих курсов, ответь максимально полезно:

        {context}

        Если подходящего курса нет, предложи альтернативные направления в IT.
        """

        # Отправляем запрос на создание ответа через DeepSeek
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Ты — эксперт по IT-курсам."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        # Логируем ответ сервера
        print("🔵 Ответ от DeepSeek (чат):", response)

        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"❌ Ошибка в ответе от DeepSeek: {response}")

    except Exception as e:
        print(f"⚠ Ошибка в get_course_info: {e}")
        return "❌ Произошла ошибка при обработке запроса."


# Пример вызова
if __name__ == "__main__":
    query = "расскажи о курсах"
    response = get_course_info(query)
    print(f"Ответ на запрос: {response}")
