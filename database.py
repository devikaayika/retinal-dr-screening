import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "retinal_history.db"


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                class_id INTEGER NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        connection.commit()


def save_prediction(image_filename, predicted_class, class_id, confidence):
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO prediction_history (
                image_filename,
                predicted_class,
                class_id,
                confidence,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            image_filename,
            predicted_class,
            class_id,
            float(confidence),
            created_at
        ))

        connection.commit()
        return cursor.lastrowid


def get_prediction_history(limit=None):
    query = """
        SELECT id, image_filename, predicted_class,
               class_id, confidence, created_at
        FROM prediction_history
        ORDER BY id DESC
    """

    parameters = ()

    if limit is not None:
        query += " LIMIT ?"
        parameters = (int(limit),)

    with get_db_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [dict(row) for row in rows]