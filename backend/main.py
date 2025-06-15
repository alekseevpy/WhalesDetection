"""
Модуль main

Инициализирует приложение FastAPI, монтирует статику и шаблоны,
определяет корневой маршрут и подключает роутер predict.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.routers.predict import router as predict_router

app = FastAPI(
    title="Whale Image Search API",
    description="Поиск похожих изображений китов на основе MegaDescriptor",
    version="1.0",
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

templates = Jinja2Templates(directory="frontend/templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """
    Обрабатывает GET /.
    Рендерит шаблон index.html для фронтенда.
    """
    return templates.TemplateResponse("index.html", {"request": request})


app.include_router(predict_router, prefix="/api")
