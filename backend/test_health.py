import pytest
import requests
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_supported_languages():
    response = client.get("/supported-languages")
    assert response.status_code == 200
    assert "languages" in response.json()

def test_translate_validation():
    response = client.post("/translate", json={
        "text": "a" * 1001,  # Exceeds max length
        "src_lang": "ne_NP",
        "tgt_lang": "en_XX"
    })
    assert response.status_code == 422  # Validation error