"""
feature_extraction.py
----------------------
MODULE 4: FEATURE EXTRACTION

Machine Learning models work with numbers, not raw text. This module
converts cleaned tweet text into numerical feature vectors using
TF-IDF (Term Frequency - Inverse Document Frequency).

Academic note (for viva):
--------------------------
TF-IDF gives every word in a tweet a weight/score:
- Words that appear often in ONE tweet but rarely across ALL tweets get
  a HIGH score (they are distinctive / important for that tweet).
- Words that appear in almost every tweet (like "the", "is") get a LOW
  score because they don't help distinguish between categories.

This is why TF-IDF is preferred over simple word counting for text
classification tasks like this one.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# ---------------------------------------------------------------------
# TF-IDF configuration
# ---------------------------------------------------------------------
# max_features : keep only the top N most informative words (keeps the
#                model simple and fast -- good for a student project).
# ngram_range  : (1, 2) means we consider both single words ("python")
#                and two-word phrases ("machine learning").
# min_df       : ignore words that appear in fewer than 2 tweets
#                (removes rare noise / typos).
# max_df       : ignore words that appear in more than 90% of tweets
#                (removes overly common, non-distinctive words).
TFIDF_PARAMS = {
    "max_features": 3000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.9,
}


def build_vectorizer():
    """Creates a new, untrained TfidfVectorizer with our chosen settings."""
    return TfidfVectorizer(**TFIDF_PARAMS)


def fit_vectorizer(vectorizer, cleaned_texts):
    """
    Fits the vectorizer on the training text and returns the
    transformed (numeric) feature matrix.
    """
    return vectorizer.fit_transform(cleaned_texts)


def save_vectorizer(vectorizer, path="models/tfidf_vectorizer.pkl"):
    """Saves the trained vectorizer to disk so it can be reused for
    predictions later, without needing to retrain."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path="models/tfidf_vectorizer.pkl"):
    """Loads a previously saved TF-IDF vectorizer from disk."""
    return joblib.load(path)
