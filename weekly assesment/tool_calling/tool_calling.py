
import os
import json
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------
# 1. Load environment variables
# ---------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

client = OpenAI(
    base_url=os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434/v1"
    ),
    api_key="ollama"
)

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


# ---------------------------------
# 2. Load customer data
# ---------------------------------

CUSTOMER_FILE = BASE_DIR / "data" / "customers.json"

with open(CUSTOMER_FILE, "r", encoding="utf-8") as file:
    customers = json.load(file)


# ---------------------------------
# 3. SQLite database path
# ---------------------------------

DB_FILE = BASE_DIR / "database" / "orders.db"


# ---------------------------------
# 4. Tool 1: Get customer
# ---------------------------------

def get_customer(customer_id):

    for customer in customers:

        if customer["customer_id"] == customer_id:

            return {
                "success": True,
                "customer": customer
            }

    return {
        "success": False,
        "error": f"Customer {customer_id} not found."
    }


# ---------------------------------
# 5. Tool 2: Get order status
# ---------------------------------

def get_order_status(order_id):

    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute(
            "SELECT * FROM orders WHERE order_id = ?",
            (order_id,)
        )

        order = cursor.fetchone()

        if order:

            return {
                "success": True,
                "order": dict(order)
            }

        return {
            "success": False,
            "error": f"Order {order_id} not found."
        }

    finally:
        connection.close()


# ---------------------------------
# 6. Available Python functions
# ---------------------------------

available_functions = {
    "get_customer": get_customer,
    "get_order_status": get_order_status
}


# ---------------------------------
# 7. Tool schemas
# ---------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Get customer details using customer ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "Customer ID"
                    }
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get order status using order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "Order ID"
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]


# ---------------------------------
# 8. Execute tool safely
# ---------------------------------

def execute_tool(function_name, arguments):

    if function_name not in available_functions:

        return {
            "success": False,
            "error": f"Unknown tool: {function_name}"
        }

    try:
        function = available_functions[function_name]

        return function(**arguments)

    except (TypeError, ValueError) as error:

        return {
            "success": False,
            "error": str(error)
        }

    except Exception as error:

        return {
            "success": False,
            "error": f"Tool execution failed: {error}"
        }


# ---------------------------------
# 9. Ask AI and handle tool calls
# ---------------------------------

def ask_question(question):

    messages = [
        {
            "role": "system",
            "content": """
            You are a customer support assistant.

            Use available tools to answer customer and order questions.

            You can call multiple tools for one question.

            Use only actual tool results.
            Never invent customer or order information.

            If a customer or order does not exist,
            clearly tell the user.
            """
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # Ask model to decide which tools to call
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # Handle multiple tool calls
    while message.tool_calls:

        messages.append(message)

        for tool_call in message.tool_calls:

            function_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            # Display clean tool call
            if function_name in available_functions:
                argument_value = next(iter(arguments.values()), "")
                print(f"Tool: {function_name}({argument_value})")
            else:
                print(f"Tool: {function_name} (unknown tool)")

            # Execute requested function
            result = execute_tool(
                function_name,
                arguments
            )

            # Return structured result to model
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        # Ask model to generate final answer
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

    return message.content


# ---------------------------------
# 10. Main program
# ---------------------------------

if __name__ == "__main__":

    question = (
        "Which plan is customer 2 on and "
        "where is order 1042?"
    )

    print("Question:", question)

    try:

        answer = ask_question(question)

        print("\nAnswer:", answer)

    except Exception as error:

        print("\nApplication error:", error)
