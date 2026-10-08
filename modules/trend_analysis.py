"""
trend_analysis.py
-------------------
MODULE 12: KEYWORD ANALYSIS
MODULE 13: HASHTAG ANALYSIS

This module extracts the most frequent/important keywords and hashtags
from a user's tweets, to support (explain) the predicted interest
category on the dashboard.
"""

import re
from collections import Counter


def extract_top_keywords(cleaned_tweets, top_n=10):
    """
    MODULE 12: KEYWORD ANALYSIS

    Counts word frequency across a list of ALREADY CLEANED tweet strings
    (stop words and punctuation already removed) and returns the top N
    most common keywords.

    Returns: list of (keyword, count) tuples, e.g.
        [("python", 8), ("ai", 6), ("machine learning", 5)]
    """
    all_words = []
    for tweet in cleaned_tweets:
        if tweet:
            all_words.extend(tweet.split())

    counter = Counter(all_words)
    return counter.most_common(top_n)


def extract_hashtags(raw_tweets):
    """
    MODULE 13: HASHTAG ANALYSIS

    Extracts hashtags (words starting with #) directly from RAW tweet
    text (before cleaning removes the '#' symbol) and counts how many
    times each hashtag appears.

    Returns: list of (hashtag, count) tuples, e.g.
        [("#AI", 5), ("#Python", 4)]
    """
    all_hashtags = []
    for tweet in raw_tweets:
        if not tweet or not isinstance(tweet, str):
            continue
        found = re.findall(r"#\w+", tweet)
        all_hashtags.extend(found)

    counter = Counter(all_hashtags)
    return counter.most_common(20)


def extract_hashtags_from_column(hashtag_strings):
    """
    Some datasets store hashtags in a dedicated 'hashtags' column
    (space-separated), rather than embedded inside the tweet text.
    This helper extracts and counts hashtags from that column instead.
    """
    all_hashtags = []
    for value in hashtag_strings:
        if not value or not isinstance(value, str):
            continue
        found = re.findall(r"#\w+", value)
        if not found:
            # Handle plain space-separated hashtags without '#' symbol.
            found = [f"#{tag.strip()}" for tag in value.split() if tag.strip()]
        all_hashtags.extend(found)

    counter = Counter(all_hashtags)
    return counter.most_common(20)
