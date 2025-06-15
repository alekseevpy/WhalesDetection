"""
Модуль routers.predict

Содержит эндпоинт /predict для классификации изображения: загрузка файла,
валидация, получение эмбеддинга и возврат топ-K похожих классов.
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.schemas.response import SearchResponse
from backend.services.model_service import get_service

router = APIRouter()


@router.post(
    "/predict",
    response_model=SearchResponse,
    summary="Классификация изображения",
    description=(
        "Принимает JPG/PNG-файл и возвращает топ-5 похожих классов "
        + "с расстояниями."
    ),
)
async def predict(
    file: UploadFile = File(..., description="Изображение (jpg, png)"),
    top_k: int = 5,
    svc=Depends(get_service),
):
    """
    Эндпоинт POST /predict.

    Аргументы:
        file (UploadFile): загружаемый файл изображения.
        top_k (int): число возвращаемых похожих классов.
        svc (ModelService): сервис модели, внедряется через Depends.
    Возвращает:
        SearchResponse: объект с результатами поиска.
    Исключения:
        HTTPException(415): неподдерживаемый формат файла.
        HTTPException(500): внутренняя ошибка сервера.
    """
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "Поддерживаются только JPEG и PNG")
    try:
        embedding = await run_in_threadpool(
            svc.preprocess_and_embed, file.file
        )
        # поиск по индексу
        results = svc.search(embedding, top_k)
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(500, f"Внутренняя ошибка: {e}") from e
