from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_ok():
    r = client.post("/chat/", json={"question": "What is asthma?"})
    assert r.status_code == 200

def test_chat_refusal():
    r = client.post("/chat/", json={"question": "Do I have asthma?"})
    assert r.status_code == 403
