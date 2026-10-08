"""
sentiment_analysis.py
-----------------------
MODULE 11: SENTIMENT ANALYSIS

This module classifies tweets as Positive, Negative, or Neutral.

Design decision (for viva):
-----------------------------
We use NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner)
because it is specifically tuned for short, informal social-media text
(slang, punctuation emphasis, emojis) and does not require training a
separate model. This keeps the project simple while still giving a
genuinely useful sentiment signal.

This module is intentionally SEPARATE from interest/future-interest
prediction. Sentiment is an additional behavioural insight, not a
replacement for the interest classifier.
"""

import nltk

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    try:
        nltk.download("vader_lexicon", quiet=True)
    except Exception:
        pass

from nltk.sentiment import SentimentIntensityAnalyzer

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def analyze_sentiment(text):
    """
    Returns a single sentiment label ('Positive', 'Negative', 'Neutral')
    for one piece of text, using VADER's compound score.
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return "Neutral"

    analyzer = _get_analyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def analyze_user_sentiment(tweets):
    """
    Runs sentiment analysis across a list of raw tweets belonging to one
    user and returns the overall percentage breakdown, e.g.:
        {"Positive": 45.0, "Negative": 15.0, "Neutral": 40.0}
    """
    if not tweets:
        return {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0}

    labels = [analyze_sentiment(t) for t in tweets]
    total = len(labels)

    result = {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0}
    for label in labels:
        result[label] += 1

    for key in result:
        result[key] = round((result[key] / total) * 100, 2)

    dominant = max(result, key=result.get)
    return {"breakdown": result, "dominant_sentiment": dominant, "labels": labels}
