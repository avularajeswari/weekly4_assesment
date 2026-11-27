
import json
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "tests" / "questions.json", "r") as file:
    questions = json.load(file)

def test_answerable_questions():
    for item in questions["answerable"]:
        response = client.post(
            "/ask",
            json={"question": item["question"]}
        )

        assert response.status_code == 200

        result = response.json()
        answer = result["answer"]

        print("\nQuestion:", item["question"])
        print("Answer:", answer)
        print("Sources:", result["sources"])

        assert item["expected"].lower() in answer.lower()
        assert len(result["sources"]) > 0

def test_unanswerable_questions():
    for item in questions["unanswerable"]:
        response = client.post(
            "/ask",
            json={"question": item["question"]}
        )

        assert response.status_code == 200

        result = response.json()

        print("\nQuestion:", item["question"])
        print("Answer:", result["answer"])

        assert result["answer"] == "I don't know."
        assert result["sources"] == []
