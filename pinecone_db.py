import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Загружаем переменные окружения из .env
load_dotenv()

# Получаем ключи из .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")  # Например, 'us-west-2'

# Создаём экземпляр клиента Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Параметры индекса
index_name = "cource"
dimension = 1536
metric = "cosine"

# Проверяем, существует ли индекс. Если нет, создаём его с использованием ServerlessSpec.
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

# Получаем объект индекса через правильный метод
index = pc.Index(index_name)

# Функция получения эмбеддинга для текста (замените её на реальную интеграцию с DeepSeek API)
def get_embedding(text):
    # Пример заглушки для тестирования: вектор с ненулевыми значениями
    return [0.1] * dimension


# Загрузка данных из JSON-файла
with open('data/course.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Формирование записей для загрузки в Pinecone
records = []
for idx, item in enumerate(data):
    # Объединяем title и description для генерации эмбеддинга
    text = f"{item['title']} {item['description']}"
    vector = get_embedding(text)
    record = {
        "id": f"record_{idx}",  # Уникальный идентификатор
        "values": vector,       # Эмбеддинг размерности 1536
        "metadata": {           # Дополнительные данные
            "title": item["title"],
            "description": item["description"]
        }
    }
    records.append(record)

# Загрузка данных батчами (если записей много)
batch_size = 100
for i in range(0, len(records), batch_size):
    batch = records[i:i + batch_size]
    index.upsert(vectors=batch)

# Вывод статистики по индексу для проверки загрузки
stats = index.describe_index_stats()
print(stats)
