def test_summarize():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.post("/summarize/", json={
        "document_text": "Chest X-ray shows mild opacity."
    })
    assert r.status_code == 200
