"""
Модуль utils.preprocess

Функция для предобработки изображений: чтение, изменение размера,
нормализация и преобразование в тензор для модели.
"""

import numpy as np
import torch
from PIL import Image


def preprocess_image(file_obj) -> torch.Tensor:
    """
    Преобразует файл изображения в тензор.

    Аргументы:
        file_obj: бинарный файловый объект с изображением.
    Возвращает:
        torch.Tensor формы (1,3,224,224).
    """
    img = Image.open(file_obj).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    tensor = (
        torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    )
    return tensor
