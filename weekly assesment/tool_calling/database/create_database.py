
import sqlite3
from pathlib import Path

# Database file location
BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "orders.db"

# Connect to SQLite database
connection = sqlite3.connect(DB_FILE)

cursor = connection.cursor()

# Create orders table
cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product TEXT,
    status TEXT
)
""")

# Insert sample orders
orders = [
    (1041, 1, "Laptop", "Shipped"),
    (1042, 2, "Mobile Phone", "Out for delivery"),
    (1043, 3, "Headphones", "Processing")
]

cursor.executemany("""
INSERT OR REPLACE INTO orders
(order_id, customer_id, product, status)
VALUES (?, ?, ?, ?)
""", orders)

connection.commit()
connection.close()

print("SQLite database created successfully!")
print("Orders inserted successfully!")
