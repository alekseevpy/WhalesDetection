"""
Модуль services.model_service

Содержит класс ModelService для загрузки PyTorch-модели, FAISS-индекса,
преобразования изображений в эмбеддинги и поиска похожих.
"""

import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import timm
import torch
import torch.nn as nn
import yaml

from backend.utils.preprocess import preprocess_image


class ModelService:
    """
    Сервис для работы с моделью и FAISS-индексом.

    Методы:
        preprocess_and_embed: преобразует изображение в эмбеддинг.
        search: ищет топ-K похожих изображений.
    """

    def __init__(self, config_path: Path):
        """
        Инициализирует сервис:
        - считывает конфигурацию из YAML,
        - загружает модель PyTorch,
        - загружает FAISS-индекс и метаданные.

        Аргументы:
            config_path (Path): путь к файлу model_config.yaml.
        """
        # Загрузка конфига
        cfg = yaml.safe_load(config_path.read_text())
        lm = cfg["load_model"]
        self.device = cfg.get("device", "cpu")

        # Загрузка модели
        if lm["compress_model"]:
            self.model = self._load_compressed(
                lm["model_name"],
                lm["compressed_shape"],
                self.device,
                lm["weights_path"],
            )
        else:
            self.model = self._load_base(
                lm["model_name"], self.device, lm["weights_path"]
            )
        self.model.eval()

        # Загрузка FAISS-индекса и метаданных
        idx_path = Path(cfg["index"]["prefix"] + ".index")
        meta_path = Path(cfg["index"]["prefix"] + ".json")

        if not idx_path.exists() or not meta_path.exists():
            raise RuntimeError(
                f"FAISS files not found: {idx_path}, {meta_path}"
            )

        self.index = faiss.read_index(str(idx_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.paths = meta["paths"]
        self.class_names = meta["class_names"]

    def _load_base(self, name, device, weights_path):
        """
        Загружает базовую модель без компрессии.

        Аргументы:
            name: имя модели в TIMM.
            device: 'cpu' или 'cuda'.
            weights_path: путь к файлу весов.
        Возвращает:
            загруженный nn.Module на указанном устройстве.
        """
        m = timm.create_model(name, pretrained=False)
        sd = torch.load(weights_path, map_location=device, weights_only=False)
        m.load_state_dict(sd, strict=True)
        return m.to(device)

    def _load_compressed(self, name, emb_size, device, weights_path):
        """
        Загружает модель с проекцией эмбеддинга на меньшую размерность.

        Аргументы:
            name: имя модели в TIMM.
            emb_size: размер выходного эмбеддинга.
            device: 'cpu' или 'cuda'.
            weights_path: путь к весам.
        Возвращает:
            nn.Module с проекцией эмбеддинга.
        """

        class Emb(nn.Module):
            """
            Внутренний класс модели с линейным слоем проекции.
            """

            def __init__(self):
                super().__init__()
                self.back = self._load_base(name, device, weights_path)
                self.proj = torch.nn.Linear(self.back.num_features, emb_size)

            def forward(self, x):
                """
                Прямой проход модели:
                - Пропускает входной тензор x через базовую модель self.back
                для получения признаков.
                - Применяет линейную проекцию self.proj для уменьшения
                размерности эмбеддинга.
                - Нормализует полученный эмбеддинг по L2-норме вдоль
                размерности признаков.

                Аргументы:
                    x (torch.Tensor): входной батч изображений (B, C, H, W).
                Возвращает:
                    torch.Tensor: L2-нормализованные эмбеддинги (B, emb_size).
                """
                f = self.back(x)
                e = self.proj(f)
                return torch.nn.functional.normalize(e, p=2, dim=1)

        emb = Emb().to(device)
        return emb

    def preprocess_and_embed(self, file_obj) -> np.ndarray:
        """
        Преобразует загруженное изображение в эмбеддинг.

        Аргументы:
            file_obj: файловый объект изображения.
        Возвращает:
            np.ndarray формы (1, D) с эмбеддингом.
        """
        tensor = preprocess_image(file_obj)  # CPU-tensor
        with torch.no_grad():
            out = self.model(tensor.to(self.device))[0]
        return out.cpu().numpy().reshape(1, -1)

    def search(self, embedding: np.ndarray, k: int) -> list:
        """
        Ищет топ-K ближайших в FAISS-индексе.

        Аргументы:
            embedding: np.ndarray формы (1, D).
            k: количество возвращаемых результатов.
        Возвращает:
            список словарей с ключами 'path', 'class_name', 'distance'.
        """
        dists, idxs = self.index.search(embedding.astype("float32"), k)
        res = []
        for dist, idx in zip(dists[0], idxs[0]):
            res.append(
                {
                    "path": self.paths[idx],
                    "class_name": self.class_names[idx],
                    "distance": float(dist),
                }
            )
        return res


@lru_cache()
def get_service():
    """
    Возвращает кешированный экземпляр ModelService.
    """
    cfg = Path(__file__).parents[2] / "model_config.yaml"
    if not cfg.exists():
        raise RuntimeError(f"Конфиг не найден по пути {cfg}")
    return ModelService(cfg)
