"""
Модуль test_preprocess
Тесты для функции preprocess_image из backend/utils/preprocess.py.
Проверяет правильность преобразования изображения в torch.Tensor.
"""

from io import BytesIO

import torch
from PIL import Image

from backend.utils.preprocess import preprocess_image


def create_test_image(color=(123, 222, 64), size=(224, 224)):
    """
    Создаёт изображение указанного цвета и размера и возвращает его как
    BytesIO.
    """
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_preprocess_image_returns_tensor():
    """
    Проверяем, что preprocess_image возвращает torch.Tensor
    с формой (1, 3, 224, 224) и значениями в диапазоне [0, 1].
    """
    buf = create_test_image()
    tensor = preprocess_image(buf)
    assert isinstance(tensor, torch.Tensor), "Ожидается объект torch.Tensor"
    assert tensor.shape == (
        1,
        3,
        224,
        224,
    ), f"Неправильная форма тензора: {tensor.shape}"
    assert torch.all(tensor >= 0) and torch.all(
        tensor <= 1
    ), "Значения тензора должны быть в [0,1]"


def test_preprocess_image_resizes_correctly():
    """
    Проверяем, что функция ресайзит изображение к размерам 224×224.
    Для этого передаём картинку другого размера и проверяем форму.
    """
    buf = create_test_image(size=(300, 150))
    tensor = preprocess_image(buf)
    assert tensor.shape[2:] == (
        224,
        224,
    ), f"Изображение не было ресайзнуто: {tensor.shape[2:]}"
