
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Connect to Ollama
client = OpenAI(
    base_url=os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434/v1"
    ),
    api_key="ollama"
)

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

DATA_FILE = BASE_DIR / "data" / "golden.jsonl"


def extract_category(response_text):
    """
    Extract only the category name from model response.
    """

    match = re.search(
        r"\b(Billing|Technical|Account)\b",
        response_text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).capitalize()

    return response_text.strip()


def run_eval(version):
    """
    Evaluate a prompt version using the golden dataset.
    """

    prompt_file = BASE_DIR / "prompts" / f"classifier_{version}.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}"
        )

    # Read prompt
    with open(prompt_file, "r", encoding="utf-8") as file:
        system_prompt = file.read()

    # Read golden dataset
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        dataset = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    correct = 0
    total = len(dataset)

    print(f"\nEvaluating Prompt {version.upper()}...")
    print("-" * 50)

    for index, item in enumerate(dataset, start=1):

        ticket = item["ticket"]
        expected = item["category"]

        # Send request to model
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.replace("{ticket}", "")
                },
                {
                    "role": "user",
                    "content": (
                        f"Customer Ticket: {ticket}\n"
                        "Return only one category: Billing, Technical, or Account."
                    )
                }
            ],
            temperature=0
        )

        # Get model response
        raw_prediction = response.choices[0].message.content.strip()

        # Normalize prediction
        predicted = extract_category(raw_prediction)

        # Compare results
        is_correct = predicted.lower() == expected.lower()

        if is_correct:
            correct += 1

        status = "PASS" if is_correct else "FAIL"

        print(
            f"Test {index:02d} | "
            f"Expected: {expected} | "
            f"Predicted: {predicted} | "
            f"{status}"
        )

    # Calculate accuracy
    accuracy = (correct / total) * 100 if total else 0

    print("-" * 50)
    print(
        f"{version.upper()}: "
        f"{correct}/{total} ({accuracy:.1f}%)"
    )

    return correct, total, accuracy


if __name__ == "__main__":

    # Evaluate V1 and V2
    v1_correct, total, v1_accuracy = run_eval("v1")
    v2_correct, _, v2_accuracy = run_eval("v2")

    # Accuracy comparison
    print("\n========== ACCURACY COMPARISON ==========")

    print(f"V1 Accuracy: {v1_accuracy:.1f}%")
    print(f"V2 Accuracy: {v2_accuracy:.1f}%")

    difference = v2_accuracy - v1_accuracy

    print(
        f"Improvement: {difference:.1f} percentage points"
    )

    # Compact summary
    print("\n========== FINAL RESULT ==========")

    print(
        f"v1: {v1_correct}/{total} ({v1_accuracy:.0f}%) | "
        f"v2: {v2_correct}/{total} ({v2_accuracy:.0f}%)"
    )
