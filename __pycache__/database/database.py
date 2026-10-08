"""
database.py
-------------
MODULE 20: DATABASE

Handles all SQLite database operations for the project.

Tables created:
    users              - registered/analyzed users
    tweets              - tweets belonging to each user (loaded from CSV)
    predictions         - future-interest prediction results (history)
    sentiment_results   - sentiment breakdown per analysis
    recommendations     - recommended topics generated per analysis

Using a dedicated database.py file (instead of writing SQL directly in
app.py) keeps the Flask routes clean and makes the database logic easy
to explain separately during a viva.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")


def get_connection():
    """Returns a new SQLite connection with row access by column name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Creates all required tables if they do not already exist.
    Safe to call every time the app starts.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            username TEXT,
            tweet TEXT NOT NULL,
            date TEXT,
            hashtags TEXT,
            likes INTEGER DEFAULT 0,
            retweets INTEGER DEFAULT 0,
            category TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            username TEXT,
            predicted_interest TEXT,
            emerging_interest TEXT,
            confidence REAL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            positive_pct REAL,
            negative_pct REAL,
            neutral_pct REAL,
            dominant_sentiment TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            category TEXT,
            recommended_topics TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def clear_tweets():
    """Removes all rows from the tweets table (used before a fresh CSV upload)."""
    conn = get_connection()
    conn.execute("DELETE FROM tweets")
    conn.commit()
    conn.close()


def insert_tweets_dataframe(df):
    """Bulk-inserts a pandas DataFrame of tweets into the tweets table."""
    conn = get_connection()
    df_to_insert = df.copy()
    expected_cols = ["user_id", "username", "tweet", "date", "hashtags", "likes", "retweets", "category"]
    for col in expected_cols:
        if col not in df_to_insert.columns:
            df_to_insert[col] = None
    df_to_insert = df_to_insert[expected_cols]
    df_to_insert.to_sql("tweets", conn, if_exists="append", index=False)
    conn.close()


def get_all_tweets_df():
    """Returns ALL tweets currently stored in the database as a DataFrame."""
    import pandas as pd
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM tweets", conn)
    conn.close()
    return df


def get_user_tweets_df(user_identifier):
    """
    Returns all tweets for a given user (matched by either user_id or
    username) as a DataFrame.
    """
    import pandas as pd
    conn = get_connection()
    query = "SELECT * FROM tweets WHERE user_id = ? OR username = ?"
    df = pd.read_sql_query(query, conn, params=(user_identifier, user_identifier))
    conn.close()
    return df


def get_distinct_users():
    """Returns a list of distinct (user_id, username) pairs available in the DB."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT DISTINCT user_id, username FROM tweets ORDER BY username"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_or_get_user(username):
    """Inserts a user if not already present, returns the users table row."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if row is None:
        cur.execute(
            "INSERT INTO users (username, created_at) VALUES (?, ?)",
            (username, datetime.now().isoformat()),
        )
        conn.commit()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
    conn.close()
    return dict(row)


def save_prediction(user_id, username, predicted_interest, emerging_interest, confidence):
    """Saves a future-interest prediction result into the history table."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO predictions
           (user_id, username, predicted_interest, emerging_interest, confidence, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, username, predicted_interest, emerging_interest, confidence,
         datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def save_sentiment_result(user_id, breakdown, dominant_sentiment):
    """Saves a sentiment analysis result into the history table."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO sentiment_results
           (user_id, positive_pct, negative_pct, neutral_pct, dominant_sentiment, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, breakdown.get("Positive", 0), breakdown.get("Negative", 0),
         breakdown.get("Neutral", 0), dominant_sentiment, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def save_recommendation(user_id, category, topics):
    """Saves the recommendation result generated for a user."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO recommendations (user_id, category, recommended_topics, created_at)
           VALUES (?, ?, ?, ?)""",
        (user_id, category, ", ".join(topics), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_prediction_history(limit=50):
    """Returns the most recent prediction history rows for display."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
