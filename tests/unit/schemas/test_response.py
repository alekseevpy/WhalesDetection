"""
Модуль tests.unit.schemas.test_response
Тесты для Pydantic-моделей SearchItem и SearchResponse.
"""

import pytest
from pydantic import ValidationError

from backend.schemas.response import SearchItem, SearchResponse


def test_search_item_valid():
    """
    Проверяем, что SearchItem корректно валидируется для правильных типов.
    """
    item = SearchItem(path="img.jpg", class_name="Whale", distance=0.123)
    assert item.path == "img.jpg"
    assert item.class_name == "Whale"
    assert isinstance(item.distance, float)


def test_search_item_invalid_types():
    """
    Проверяем, что ValidationError возникает при некорректных типах полей.
    """
    with pytest.raises(ValidationError):
        SearchItem(path=123, class_name="Whale", distance=0.123)
    with pytest.raises(ValidationError):
        SearchItem(path="img.jpg", class_name=456, distance=0.123)
    with pytest.raises(ValidationError):
        SearchItem(path="img.jpg", class_name="Whale", distance="bad")


def test_search_response_valid_list():
    """
    Проверяем, что SearchResponse принимает список моделей SearchItem.
    """
    item = SearchItem(path="img.jpg", class_name="W", distance=1.0)
    resp = SearchResponse(results=[item])
    assert isinstance(resp.results, list)
    assert isinstance(resp.results[0], SearchItem)


def test_search_response_invalid_list_type():
    """
    Проверяем, что ValidationError возникает при передаче не списка в results.
    """
    with pytest.raises(ValidationError):
        SearchResponse(results="not a list")
