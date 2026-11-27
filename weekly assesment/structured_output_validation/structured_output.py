
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


# Load environment variables
load_dotenv()

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


# Pydantic model
class TicketInfo(BaseModel):
    customer_name: str
    product: str
    issue_type: str
    urgency: int = Field(ge=1, le=5)
    summary: str


# File paths
DATA_FILE = BASE_DIR / "data" / "tickets.json"
OUTPUT_FILE = BASE_DIR / "outputs" / "results.json"


def load_tickets():
    """Load customer tickets from JSON file."""

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_ticket_info(ticket, demonstrate_retry=False):
    """Extract structured information with validation and retry."""

    messages = [
        {
            "role": "system",
            "content": """
            Extract information from the customer ticket.

            Return ONLY valid JSON with these fields:
            customer_name, product, issue_type, urgency, summary.

            urgency must be an integer between 1 and 5.
            Do not include markdown or extra text.
            """
        },
        {
            "role": "user",
            "content": f"Customer ticket: {ticket}"
        }
    ]

    for attempt in range(1, 4):

        print(f"\nAttempt {attempt}:")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"}
            )

            raw_json = response.choices[0].message.content

            # Intentionally simulate invalid urgency for first ticket
            if attempt == 1 and demonstrate_retry:
                test_data = json.loads(raw_json)
                test_data["urgency"] = 7
                raw_json = json.dumps(test_data)

            # Validate JSON using Pydantic
            result = TicketInfo.model_validate_json(raw_json)

            print("Success!")
            return {
                "status": "success",
                "attempts": attempt,
                "data": result.model_dump()
            }

        except ValidationError as error:

            print("Validation error!")
            print(error)

            messages.append({
                "role": "assistant",
                "content": raw_json
            })

            messages.append({
                "role": "user",
                "content": (
                    "Your previous JSON failed validation. "
                    "Fix these errors and return valid JSON only:\n"
                    f"{error}"
                )
            })

        except Exception as error:
            print(f"Error: {error}")

            return {
                "status": "failed",
                "attempts": attempt,
                "error": str(error)
            }

    print("Failed after 3 attempts.")

    return {
        "status": "failed",
        "attempts": 3,
        "error": "Validation failed after 3 attempts."
    }


if __name__ == "__main__":

    tickets = load_tickets()

    print(f"Total tickets loaded: {len(tickets)}")

    all_results = []

    for index, item in enumerate(tickets, start=1):

        print("\n" + "=" * 50)
        print(f"Processing Ticket {index}")
        print("=" * 50)

        result = extract_ticket_info(
            item["ticket"],
            demonstrate_retry=(index == 1)
        )

        all_results.append({
            "ticket": item["ticket"],
            "result": result
        })

    # Save results to JSON file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4)

    print("\n" + "=" * 50)
    print("Processing completed!")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 50)
