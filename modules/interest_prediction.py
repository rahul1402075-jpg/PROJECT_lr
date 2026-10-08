"""
interest_prediction.py
------------------------
MODULE 8: INTEREST CLASSIFICATION
MODULE 9: FUTURE INTEREST PREDICTION
MODULE 10: INTEREST TREND ANALYSIS

This is the "brain" of the project. It uses the trained ML model to:
1. Classify individual tweets into interest categories.
2. Aggregate many tweets from one user into an overall interest profile.
3. Compare recent activity vs older activity to detect emerging interests.
4. Produce a clearly-labeled "future interest prediction".

IMPORTANT (Academic honesty):
This module predicts likely FUTURE interest based on patterns in PAST
behaviour. It is a probabilistic estimate, not a guarantee, and this is
made explicit in every result returned by this module.
"""

import numpy as np
import pandas as pd


def predict_category(tweet_text, model, vectorizer, clean_text_fn):
    """
    MODULE 8: INTEREST CLASSIFICATION

    Predicts the interest category of a single tweet and returns the
    predicted label along with a confidence score (probability) when
    the underlying model supports it (Logistic Regression / Naive Bayes).
    For models without probability support (e.g. plain LinearSVC), we
    fall back to the decision function margin, normalized to look like
    a confidence percentage.
    """
    cleaned = clean_text_fn(tweet_text)
    if not cleaned:
        return {"category": "Unknown", "confidence": 0.0}

    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities)) * 100
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features)[0]
        # Convert raw SVM margins into a pseudo-probability using softmax,
        # purely for a friendlier display value (0-100%).
        scores = np.atleast_1d(scores)
        exp_scores = np.exp(scores - np.max(scores))
        softmax = exp_scores / exp_scores.sum()
        confidence = float(np.max(softmax)) * 100
    else:
        confidence = 100.0  # Model gives no confidence information

    return {"category": prediction, "confidence": round(confidence, 2)}


def analyze_user_tweets(df_user, model, vectorizer, clean_text_fn):
    """
    MODULE 9: FUTURE INTEREST PREDICTION

    Takes ALL historical tweets belonging to a single user and:
    1. Predicts a category for every tweet.
    2. Calculates the frequency (%) of each category -> "current interests".
    3. Splits tweets into an "older half" and "recent half" (using the
       tweet date, if available) to detect increasing/decreasing interest.
    4. Determines a Primary Future Interest (highest overall frequency)
       and a Potential Emerging Interest (category growing fastest in the
       recent half compared to the older half).

    Returns a dictionary with everything the dashboard needs.
    """
    if df_user.empty:
        return None

    df_user = df_user.copy()

    # Step 1: predict category for every tweet the user has posted.
    predicted_categories = []
    confidences = []
    for tweet in df_user["tweet"]:
        result = predict_category(tweet, model, vectorizer, clean_text_fn)
        predicted_categories.append(result["category"])
        confidences.append(result["confidence"])

    df_user["predicted_category"] = predicted_categories
    df_user["confidence"] = confidences

    # Step 2: overall frequency of each category = "current interests".
    total_tweets = len(df_user)
    category_counts = df_user["predicted_category"].value_counts()
    current_interests = (category_counts / total_tweets * 100).round(2).to_dict()

    # Step 3: split into older half vs recent half using date if available.
    trend_info = {}
    if "date" in df_user.columns:
        try:
            df_user["date"] = pd.to_datetime(df_user["date"], errors="coerce")
            df_sorted = df_user.dropna(subset=["date"]).sort_values("date")
        except Exception:
            df_sorted = df_user
    else:
        df_sorted = df_user

    if len(df_sorted) >= 4:
        midpoint = len(df_sorted) // 2
        older_half = df_sorted.iloc[:midpoint]
        recent_half = df_sorted.iloc[midpoint:]

        older_dist = older_half["predicted_category"].value_counts(normalize=True) * 100
        recent_dist = recent_half["predicted_category"].value_counts(normalize=True) * 100

        all_categories = set(older_dist.index) | set(recent_dist.index)
        for cat in all_categories:
            old_pct = round(older_dist.get(cat, 0.0), 2)
            new_pct = round(recent_dist.get(cat, 0.0), 2)
            change = round(new_pct - old_pct, 2)
            if change > 5:
                status = "increasing"
            elif change < -5:
                status = "decreasing"
            else:
                status = "stable"
            trend_info[cat] = {"older_pct": old_pct, "recent_pct": new_pct,
                                "change": change, "status": status}
    else:
        # Not enough data to split into two halves reliably.
        for cat, pct in current_interests.items():
            trend_info[cat] = {"older_pct": pct, "recent_pct": pct,
                                "change": 0.0, "status": "stable"}

    # Step 4: Primary future interest = highest overall frequency category.
    primary_future_interest = category_counts.idxmax() if not category_counts.empty else None

    # Emerging interest = category with the biggest POSITIVE change in the
    # recent half that is NOT already the primary interest.
    emerging_candidates = {
        cat: info["change"] for cat, info in trend_info.items()
        if cat != primary_future_interest
    }
    potential_emerging_interest = None
    if emerging_candidates:
        best_cat = max(emerging_candidates, key=emerging_candidates.get)
        if emerging_candidates[best_cat] > 0:
            potential_emerging_interest = best_cat

    avg_confidence = round(float(np.mean(confidences)), 2) if confidences else 0.0

    return {
        "total_tweets": total_tweets,
        "current_interests": current_interests,          # dict: category -> %
        "trend_info": trend_info,                         # dict: category -> {older_pct, recent_pct, change, status}
        "primary_future_interest": primary_future_interest,
        "potential_emerging_interest": potential_emerging_interest,
        "average_confidence": avg_confidence,
        "disclaimer": (
            "This is a prediction based on the user's historical tweet "
            "patterns. It is a probabilistic estimate, not a guaranteed "
            "future behaviour."
        ),
        "detailed_predictions": df_user[["tweet", "date", "predicted_category", "confidence"]]
        .assign(date=lambda d: d["date"].astype(str))
        .to_dict(orient="records"),
    }


def monthly_trend(df_user):
    """
    MODULE 10: INTEREST TREND ANALYSIS

    Groups a user's predicted tweets by month and calculates what
    percentage of that month's tweets belonged to each category.
    Used to draw the "Interest Trends" line chart on the dashboard.
    """
    if "predicted_category" not in df_user.columns or "date" not in df_user.columns:
        return {}

    df = df_user.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return {}

    df["month"] = df["date"].dt.strftime("%Y-%m")
    grouped = df.groupby(["month", "predicted_category"]).size().unstack(fill_value=0)
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    percentages = percentages.round(2).sort_index()

    return {
        "months": list(percentages.index),
        "categories": {col: list(percentages[col]) for col in percentages.columns},
    }
