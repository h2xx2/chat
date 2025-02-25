import openai
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Инициализация Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
index = pinecone.Index("courses")  # Имя индекса


def get_course_info(query):
    """Ищет похожие курсы в Pinecone и передает их в OpenAI."""

    # Генерируем эмбеддинг для запроса пользователя
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY
    )
    vector = response["data"][0]["embedding"]

    # Ищем похожие курсы
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

    # Запрашиваем OpenAI
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Ты — эксперт по IT-курсам."},
                  {"role": "user", "content": prompt}],
        api_key=OPENAI_API_KEY
    )

    return gpt_response["choices"][0]["message"]["content"]
