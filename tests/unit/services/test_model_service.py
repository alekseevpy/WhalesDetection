"""
Модуль tests.unit.services.test_model_service

Тесты для класса ModelService:
    - загрузка базовой модели из конфига
    - preprocess_and_embed корректно вызывает preprocess_image и возвращает
    numpy.ndarray
    - search возвращает топ-K результатов из FAISS-индекса
"""

import json

import faiss
import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from backend.services.model_service import ModelService


@pytest.fixture
def config_and_index(tmp_path, monkeypatch):
    """
    Фикстура для создания временного конфига, весов модели и FAISS-индекса.

    Возвращает:
        Tuple[Path, int]: путь к config.yaml и размерность эмбеддинга.
    """
    base_dim = 4

    class DummyModel(nn.Module):
        """
        Простая линейная модель для тестирования загрузки.
        """

        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(base_dim, base_dim)

        def forward(self, x):
            """
            Прямой проход входного тензора через линейный слой.
            """
            return self.layer(x)

    # сохраняем state_dict DummyModel
    weights_path = tmp_path / "weights.pth"
    dummy = DummyModel()
    torch.save(dummy.state_dict(), weights_path)

    # подменяем timm.create_model и torch.load
    monkeypatch.setattr(
        "timm.create_model", lambda name, pretrained: DummyModel()
    )
    monkeypatch.setattr(
        "torch.load",
        lambda path, map_location=None, **kwargs: dummy.state_dict(),
    )

    # пишем конфиг YAML
    cfg = {
        "load_model": {
            "compress_model": False,
            "model_name": "dummy",
            "compressed_shape": base_dim,
            "weights_path": str(weights_path),
        },
        "index": {"prefix": str(tmp_path / "test_index")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg))

    # создаём FAISS-индекс и метаданные
    embeddings = np.eye(base_dim, dtype="float32")
    index = faiss.IndexFlatL2(base_dim)
    index.add(embeddings)
    idx_file = tmp_path / "test_index.index"
    faiss.write_index(index, str(idx_file))

    meta = {
        "paths": [f"p{i}" for i in range(base_dim)],
        "class_names": [f"c{i}" for i in range(base_dim)],
    }
    meta_path = tmp_path / "test_index.json"
    meta_path.write_text(json.dumps(meta))

    return config_path, base_dim


def test_preprocess_and_embed(monkeypatch, config_and_index):
    """
    Проверяет, что preprocess_and_embed возвращает numpy.ndarray
    правильной формы.
    Мы мокируем preprocess_image и затем переназначаем svc.model,
    чтобы не было несоответствия форм.
    """
    config_path, base_dim = config_and_index

    # Подменяем функцию preprocess_image на выдачу тензора (1, base_dim)
    fake_tensor = torch.zeros(1, base_dim)
    monkeypatch.setattr(
        "backend.services.model_service.preprocess_image",
        lambda f: fake_tensor,
    )

    svc = ModelService(config_path)
    # Переназначаем модель: на вход берёт любой тензор,
    # возвращает fake_tensor
    svc.model = lambda x: fake_tensor

    emb = svc.preprocess_and_embed(None)

    assert isinstance(emb, np.ndarray), "Ожидается numpy.ndarray"
    assert emb.shape == (
        1,
        base_dim,
    ), f"Неверная форма эмбеддинга: {emb.shape}"


def test_search_returns_top_k(config_and_index):
    """
    Проверяет, что метод search возвращает корректный топ-K результатов.
    Используем нулевой запрос, чтобы ближайшим оказался индекс 0.
    """
    config_path, base_dim = config_and_index
    svc = ModelService(config_path)

    query = np.zeros((1, base_dim), dtype="float32")
    k = 3
    results = svc.search(query, k)

    assert isinstance(results, list)
    assert len(results) == k

    for i, item in enumerate(results):
        assert item["path"] == f"p{i}"
        assert item["class_name"] == f"c{i}"
        assert isinstance(item["distance"], float)
