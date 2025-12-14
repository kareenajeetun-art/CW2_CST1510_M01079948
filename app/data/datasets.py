import pandas as pd
from pathlib import Path
from app.data.db import connect_database
from app.data.datasets import insert_dataset_metadata


def insert_dataset_metadata(dataset_name, category, source, last_updated,
                            record_count=None, file_size_mb=None):
    """Insert new dataset metadata record."""
    
    conn = connect_database()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO datasets_metadata
        (dataset_name, category, source, last_updated,
         record_count, file_size_mb)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (dataset_name, category, source, last_updated,
          record_count, file_size_mb))

    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id



def get_all_datasets_metadata():
    """Return all dataset metadata records as a DataFrame."""
    conn = connect_database()
    df = pd.read_sql_query(
        "SELECT * FROM datasets_metadata ORDER BY id DESC",
        conn
    )
    conn.close()
    return df


def update_dataset_metadata(conn, dataset_id, field, new_value):
    """
    Update a single field of a dataset metadata record.
    Example: update_dataset_metadata(conn, 3, "source", "admin_team")
    """
    cursor = conn.cursor()

    query = f"UPDATE datasets_metadata SET {field} = ? WHERE id = ?"
    cursor.execute(query, (new_value, dataset_id))

    conn.commit()
    return cursor.rowcount



def delete_dataset_metadata(conn, dataset_id):
    """Delete a dataset metadata record by ID."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM datasets_metadata WHERE id = ?", (dataset_id,))
    conn.commit()
    return cursor.rowcount



def count_datasets_by_category(conn):
    """Count datasets grouped by category."""
    query = """
    SELECT category, COUNT(*) AS count
    FROM datasets_metadata
    GROUP BY category
    ORDER BY count DESC
    """
    return pd.read_sql_query(query, conn)


def count_large_datasets(conn, min_rows=100000):
    """Return datasets with record_count greater than min_rows."""
    query = """
    SELECT dataset_name, record_count
    FROM datasets_metadata
    WHERE record_count > ?
    ORDER BY record_count DESC
    """
    return pd.read_sql_query(query, conn, params=(min_rows,))


def datasets_recently_updated(conn, days=90):
    """Return datasets updated within last X days."""
    query = f"""
    SELECT dataset_name, last_updated
    FROM datasets_metadata
    WHERE DATE(last_updated) >= DATE('now', '-{days} days')
    ORDER BY last_updated DESC
    """
    return pd.read_sql_query(query, conn)

import sqlite3
from pathlib import Path

DB_PATH = Path("DATA/intelligence_platform.db")


def insert_dataset_metadata(
    name: str,
    source: str,
    description: str,
    owner: str,
    created_date: str
):
    """
    Insert a new dataset metadata record into the database.
    """

    if not DB_PATH.exists():
        raise FileNotFoundError("Database file not found.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO datasets_metadata
        (name, source, description, owner, created_date)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, source, description, owner, created_date)
    )

    conn.commit()
    conn.close()

def load_csv_to_table_datasets_metadata(conn, csv_path, table_name):
    """Load dataset metadata CSV into the database."""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"⚠️ File not found: {csv_path}")
        return 0

    df = pd.read_csv(csv_path)

    # Rename CSV → DB schema
    df = df.rename(columns={
        "name": "dataset_name",
        "rows": "record_count",
        "uploaded_by": "source",
        "upload_date": "last_updated"
    })

    # Add missing required fields
    if "category" not in df.columns:
        df["category"] = "General"

    if "file_size_mb" not in df.columns:
        df["file_size_mb"] = None

    # Only keep columns matching DB schema
    required_cols = [
        "dataset_name", "category", "source",
        "last_updated", "record_count", "file_size_mb"
    ]

    df = df[required_cols]

    # Load into database
    df.to_sql(
        name=table_name,
        con=conn,
        if_exists="append",
        index=False
    )

    print(f"✅ Loaded {len(df)} rows into '{table_name}'.")
    return len(df)
