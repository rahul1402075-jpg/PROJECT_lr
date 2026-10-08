"""
train_model.py
----------------
MODULE 5: MACHINE LEARNING
MODULE 6: MODEL TRAINING
MODULE 7: MODEL EVALUATION

This script:
    1. Loads the dataset (CSV).
    2. Validates required columns.
    3. Cleans tweets using the shared clean_text() function.
    4. Removes invalid/empty records.
    5. Extracts TF-IDF features.
    6. Splits data into train/test sets (80/20, stratified).
    7. Trains Logistic Regression (primary model), Naive Bayes, and SVM.
    8. Evaluates all three using Accuracy, Precision, Recall, F1-score,
       and a Confusion Matrix.
    9. Automatically selects the best-performing model (highest F1-score).
    10. Saves the best model, the TF-IDF vectorizer, and the metrics.

Run this script whenever you want to (re)train the model, e.g. after
uploading a new dataset:

    python train_model.py

Outputs:
    models/best_model.pkl
    models/tfidf_vectorizer.pkl
    models/model_metrics.json
"""

import os
import sys
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Make sure local packages (preprocessing/) can be imported regardless of
# where this script is run from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.clean_text import clean_text, is_valid_tweet
from preprocessing.feature_extraction import (
    build_vectorizer, fit_vectorizer, save_vectorizer, TFIDF_PARAMS
)

REQUIRED_COLUMNS = ["user_id", "username", "tweet", "category"]
DATASET_PATH = os.path.join("dataset", "twitter_dataset.csv")
MODELS_DIR = "models"


def load_dataset(path):
    """MODULE 6, Step 1-2: Load the dataset and validate required columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please upload a CSV dataset first "
            f"(see MODULE 1: DATA COLLECTION)."
        )

    df = pd.read_csv(path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required column(s): {missing_cols}. "
            f"Required columns are: {REQUIRED_COLUMNS}"
        )

    if df.empty:
        raise ValueError("Dataset is empty. Please provide a dataset with at least a few rows.")

    return df


def preprocess_dataset(df):
    """MODULE 6, Steps 3-4: Clean tweets and remove invalid records."""
    df = df.copy()
    df["tweet"] = df["tweet"].astype(str)
    df["cleaned_tweet"] = df["tweet"].apply(clean_text)

    # Remove rows with null/empty/too-short tweets or missing category labels.
    valid_mask = df.apply(
        lambda row: is_valid_tweet(row["tweet"], row["cleaned_tweet"]) and pd.notna(row["category"]),
        axis=1,
    )
    df = df[valid_mask].reset_index(drop=True)

    # Remove exact duplicate tweets FROM THE SAME USER (keep the first
    # occurrence). Two different users posting similar text is still
    # valid, distinct training data, so we only dedupe within a user.
    df = df.drop_duplicates(subset=["user_id", "cleaned_tweet"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid tweets remained after cleaning. Please check your dataset "
            "for empty or extremely short tweet text."
        )

    return df


def evaluate_model(model, X_test, y_test, model_name):
    """MODULE 7: MODEL EVALUATION - computes real metrics from predictions."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"\n{model_name}:")
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall   : {recall * 100:.2f}%")
    print(f"  F1 Score : {f1 * 100:.2f}%")

    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def main():
    print("=" * 60)
    print("STEP 1-2: Loading and validating dataset")
    print("=" * 60)
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df)} rows from {DATASET_PATH}")

    print("\n" + "=" * 60)
    print("STEP 3-4: Cleaning tweets and removing invalid records")
    print("=" * 60)
    df = preprocess_dataset(df)
    print(f"{len(df)} valid tweets remain after cleaning")

    print("\n" + "=" * 60)
    print("STEP 5: TF-IDF Feature Extraction")
    print("=" * 60)
    vectorizer = build_vectorizer()
    X = fit_vectorizer(vectorizer, df["cleaned_tweet"])
    y = df["category"]
    print(f"TF-IDF matrix shape: {X.shape} (parameters: {TFIDF_PARAMS})")

    print("\n" + "=" * 60)
    print("STEP 6: Train/Test Split (80/20, stratified where possible)")
    print("=" * 60)
    # Stratification requires at least 2 samples per class; fall back to a
    # plain split if the dataset is too small/imbalanced for that.
    class_counts = y.value_counts()
    can_stratify = (class_counts.min() >= 2) and (len(class_counts) > 1)
    stratify_arg = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )
    print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

    print("\n" + "=" * 60)
    print("STEP 7-9: Training Logistic Regression, Naive Bayes, and SVM")
    print("=" * 60)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(max_iter=5000, random_state=42),
    }

    metrics = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics[name] = evaluate_model(model, X_test, y_test, name)

    print("\n" + "=" * 60)
    print("STEP 10-12: Comparing models and selecting the best one")
    print("=" * 60)
    # Best model = highest F1-score (a balanced measure of precision & recall).
    best_model_name = max(metrics, key=lambda name: metrics[name]["f1_score"])
    best_model = trained_models[best_model_name]
    print(f"Best performing model: {best_model_name} "
          f"(F1 Score: {metrics[best_model_name]['f1_score']}%)")

    print("\n" + "=" * 60)
    print("STEP 13-15: Saving best model, vectorizer, and metrics")
    print("=" * 60)
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    save_vectorizer(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

    metrics_output = {
        "best_model": best_model_name,
        "models": metrics,
        "training_samples": int(X_train.shape[0]),
        "testing_samples": int(X_test.shape[0]),
        "total_samples": int(X.shape[0]),
        "categories": sorted(y.unique().tolist()),
    }
    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics_output, f, indent=4)

    print(f"\nSaved: {MODELS_DIR}/best_model.pkl")
    print(f"Saved: {MODELS_DIR}/tfidf_vectorizer.pkl")
    print(f"Saved: {MODELS_DIR}/model_metrics.json")
    print("\nModel training complete!")


if __name__ == "__main__":
    main()
