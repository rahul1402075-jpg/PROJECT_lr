"""
clean_text.py
-------------
MODULE 2: DATA PREPROCESSING
MODULE 3: TOKENIZATION

This file contains the core NLP preprocessing pipeline used throughout the
project. Every raw tweet passes through the `clean_text()` function before
it is used for training or prediction.

Academic note (for viva):
--------------------------
Data Mining projects that work with text always need a cleaning step
because raw text contains a lot of "noise" (URLs, emojis, punctuation,
stop words) that do not help a classifier learn meaningful patterns.
Removing this noise and reducing words to their base form (lemmatization)
lets the model focus on the words that actually carry meaning.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------
# Download required NLTK resources.
# These downloads only happen once; after that NLTK caches them locally.
# We wrap this in try/except blocks so the app does not crash if the
# resources are already present.
# ---------------------------------------------------------------------
def _ensure_nltk_resources():
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                # If download fails (e.g. no internet), we continue.
                # The functions below have fallbacks that do not
                # strictly require every resource.
                pass


_ensure_nltk_resources()

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    # Minimal fallback stopword list in case NLTK data is unavailable.
    STOP_WORDS = {
        "a", "an", "the", "is", "am", "are", "was", "were", "i", "you",
        "he", "she", "it", "we", "they", "and", "or", "but", "to", "of",
        "in", "on", "for", "with", "this", "that", "at", "as", "be",
    }

lemmatizer = WordNetLemmatizer()


def tokenize_text(text: str):
    """
    MODULE 3: TOKENIZATION

    Splits a sentence into a list of individual words (tokens).

    Example:
        "I love Artificial Intelligence"
        -> ["I", "love", "Artificial", "Intelligence"]
    """
    if not text:
        return []
    try:
        return word_tokenize(text)
    except Exception:
        # Fallback simple tokenizer if NLTK punkt data is unavailable.
        return text.split()


def clean_text(text):
    """
    MODULE 2: DATA PREPROCESSING

    Cleans a single raw tweet and returns a clean, lemmatized string
    ready to be used for TF-IDF feature extraction.

    Steps performed (in order):
        1. Handle null / empty / non-string input safely.
        2. Lowercase the text.
        3. Remove URLs.
        4. Remove @mentions.
        5. Remove hashtag symbol (# ) but keep the word after it.
        6. Remove punctuation.
        7. Remove numbers.
        8. Remove emojis / non-ASCII symbols.
        9. Remove extra whitespace.
        10. Tokenize.
        11. Remove stop words.
        12. Lemmatize each remaining token.

    Example:
        Input:  "@OpenAI I really love #AI and Machine Learning! 🚀 https://example.com"
        Output: "really love ai machine learning"
    """
    # 1. Handle null / missing / non-string values safely.
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if len(text) == 0:
        return ""

    # 2. Lowercase
    text = text.lower()

    # 3. Remove URLs (http://, https://, www.)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 4. Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # 5. Remove the '#' symbol but keep the hashtag word itself
    #    (e.g. "#AI" -> "AI") so the topic word can still help classification.
    text = re.sub(r"#", " ", text)

    # 6. Remove emojis and other non-ASCII characters.
    text = text.encode("ascii", "ignore").decode("ascii")

    # 7. Remove punctuation.
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 8. Remove numbers (digits).
    text = re.sub(r"\d+", " ", text)

    # 9. Remove extra whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) == 0:
        return ""

    # 10. Tokenize.
    tokens = tokenize_text(text)

    # 11. Remove stop words and very short tokens (like leftover single letters),
    #     then 12. Lemmatize each token to its base/dictionary form.
    cleaned_tokens = []
    for token in tokens:
        if token in STOP_WORDS:
            continue
        if len(token) <= 1:
            continue
        lemma = lemmatizer.lemmatize(token)
        cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


def is_valid_tweet(raw_text, cleaned_text, min_length=2):
    """
    Helper used during dataset loading to filter out tweets that are:
    - null / empty
    - too short to carry any real meaning after cleaning
    This keeps the training data high quality.
    """
    if raw_text is None:
        return False
    if cleaned_text is None or len(cleaned_text.strip()) == 0:
        return False
    if len(cleaned_text.split()) < min_length:
        return False
    return True


if __name__ == "__main__":
    # Small self-test / demonstration when this file is run directly.
    sample = "@OpenAI I really love #AI and Machine Learning! 🚀 https://example.com"
    print("Input :", sample)
    print("Output:", clean_text(sample))
