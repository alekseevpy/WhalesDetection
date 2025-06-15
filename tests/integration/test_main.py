"""
Модуль tests.integration.test_main

Интеграционные тесты для главного маршрута и статических файлов:
  - GET / возвращает HTML-страницу с корректным заголовком
  - статика отдается из /static
"""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_index_page_returns_html():
    """
    Проверяем, что GET / возвращает HTML со статусом 200
    и содержит тег <title>WhalesDetection</title>.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>WhalesDetection</title>" in response.text


def test_static_css_served():
    """
    Проверяем, что статический CSS-файл отдается корректно:
    статус 200, content-type содержит text/css,
    и в тексте есть селектор body или .header.
    """
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    ctype = response.headers.get("content-type", "")
    assert "text/css" in ctype
    text = response.text.strip()
    assert text.startswith("body") or ".header" in text
