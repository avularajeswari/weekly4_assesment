
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    api_key="ollama"
)

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


CLASSIFICATION_PROMPT = """
You are a customer support ticket classifier.

Classify the given ticket into exactly one category:
1. Billing
2. Technical
3. Account

Return only the category name.
Do not provide any explanation.

Customer Ticket:
{ticket}
"""


def classify_ticket(ticket: str):
    prompt = CLASSIFICATION_PROMPT.format(ticket=ticket)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    category = response.choices[0].message.content.strip()

    allowed_categories = ["Billing", "Technical", "Account"]

    if category not in allowed_categories:
        return "Technical"

    return category
