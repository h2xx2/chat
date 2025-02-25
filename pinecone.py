import pinecone
import json
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# API-ключи
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Инициализация Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index("cource")  # Имя индекса

# Функция для загрузки данных в Pinecone
def load_courses_to_pinecone():
    with open("data/course.json", "r", encoding="utf-8") as file:
        courses = json.load(file)

    for course in courses:
        title = course["title"]
        description = course["description"]
        text = f"{title}. {description}"  # Объединяем для векторизации

        # Создаем эмбеддинг через OpenAI
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002",
            api_key=OPENAI_API_KEY
        )
        vector = response["data"][0]["embedding"]

        # Загружаем в Pinecone
        index.upsert([(title, vector, {"description": description})])

    print("Данные успешно загружены в Pinecone!")

if __name__ == "__main__":
    load_courses_to_pinecone()
