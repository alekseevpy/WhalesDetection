"""
Модуль tests.integration.test_predict

Интеграционные тесты для эндпоинта /api/predict:
  - успешный запрос возвращает список длины top_k
  - неподдерживаемый формат файла возвращает 415
  - отсутствие файла возвращает 422
  - top_k=0 возвращает пустой список
"""

from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.services.model_service import get_service

client = TestClient(app)


class DummyService:
    """
    Заглушка сервиса модели для интеграционных тестов.
    """

    def preprocess_and_embed(self, file_obj):
        """
        Возвращаем нулевой эмбеддинг размерности 128.
        """
        return np.zeros((1, 128), dtype="float32")

    def search(self, embedding, k):
        """
        Возвращаем k фиктивных результатов.
        """
        return [
            {"path": f"p{i}", "class_name": f"c{i}", "distance": float(i)}
            for i in range(k)
        ]


@pytest.fixture(autouse=True)
def override_service():
    """
    Подменяем зависимость get_service на DummyService через
    dependency_overrides.
    """
    app.dependency_overrides[get_service] = lambda: DummyService()
    yield
    app.dependency_overrides.clear()  # очищаем после теста


def create_image_bytes():
    """
    Создаёт картинку JPEG 224×224 и возвращает её как BytesIO.
    """
    img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_predict_success():
    """
    Успешный POST /api/predict возвращает top_k результатов.
    """
    buf = create_image_bytes()
    response = client.post(
        "/api/predict?top_k=3",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 3
    for i, item in enumerate(data["results"]):
        assert item["path"] == f"p{i}"
        assert item["class_name"] == f"c{i}"
        assert isinstance(item["distance"], float)


def test_predict_unsupported_format():
    """
    Отправка текстового файла должна вернуть 415.
    """
    response = client.post(
        "/api/predict",
        files={"file": ("test.txt", BytesIO(b"abc"), "text/plain")},
    )
    assert response.status_code == 415


def test_predict_missing_file():
    """
    Отсутствие файла в запросе даёт 422 Validation Error.
    """
    response = client.post("/api/predict")
    assert response.status_code == 422


def test_predict_zero_top_k():
    """
    Запрос с top_k=0 должен вернуть пустой список.
    """
    buf = create_image_bytes()
    response = client.post(
        "/api/predict?top_k=0",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []
