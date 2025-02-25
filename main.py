from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from openAI import get_course_info

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket API для чата
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        query = await websocket.receive_text()  # Получаем запрос от клиента
        answer = get_course_info(query)  # Обрабатываем через OpenAI
        await websocket.send_text(answer)  # Отправляем ответ
