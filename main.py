from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

from openAI import get_course_info

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешает все источники
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket API для чата
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            query = await websocket.receive_text()
            print(f"Получен запрос: {query}")
            answer = get_course_info(query)
            await websocket.send_text(answer)
    except WebSocketDisconnect:
        print("Клиент отключился")
