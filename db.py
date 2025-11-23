import psycopg2
import os
from psycopg2.extras import register_uuid

# Register UUID adapter
register_uuid()

DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", 5432),
    "dbname": os.getenv("PG_DB", "postgres"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "pass"),
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)
