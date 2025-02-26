import openai
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# Инициализация Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Индекс для работы
index_name = "cource"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Размер эмбеддингов
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

# Используем правильное создание индекса
index = pc.Index(index_name)

def get_course_info(query):
    """Ищет похожие курсы с помощью Pinecone и передает их в OpenAI."""

    # Генерация эмбеддингов с использованием нового API
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Используем модель для эмбеддингов
        input=query
    )
    vector = response['data'][0]['embedding']  # Извлекаем эмбеддинг из ответа

    # Ищем похожие курсы с помощью Pinecone
    results = index.query(vector, top_k=3, include_metadata=True)

    # Формируем контекст для OpenAI
    context = "\n\n".join(
        [f"Название: {match['id']}\nОписание: {match['metadata']['description']}" for match in results["matches"]])

    # Промпт для GPT
    prompt = f"""
Ты — консультант по обучению. Пользователь спрашивает: "{query}". 
На основе следующих курсов, ответь максимально полезно:

{context}

Если подходящего курса нет, предложи альтернативные направления в IT.
"""

    # Запрашиваем OpenAI для создания ответа на основе контекста
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Ты — эксперт по IT-курсам."},
                  {"role": "user", "content": prompt}],
        api_key=OPENAI_API_KEY
    )

    return gpt_response["choices"][0]["message"]["content"]
