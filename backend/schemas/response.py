"""
Модуль schemas.response

Определяет Pydantic-модели для ответа эндпоинта /predict.
"""

from typing import List

from pydantic import BaseModel


class SearchItem(BaseModel):
    """
    Описывает один результат поиска.

    Атрибуты:
        path (str): путь к изображению-результату.
        class_name (str): название класса изображения.
        distance (float): расстояние в эмбеддинговом пространстве.
    """

    path: str
    class_name: str
    distance: float


class SearchResponse(BaseModel):
    """
    Модель ответа для /predict.

    Атрибуты:
        results (List[SearchItem]): список результатов поиска.
    """

    results: List[SearchItem]
