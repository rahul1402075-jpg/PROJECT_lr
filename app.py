"""
app.py
-------
MODULE 15: WEB APPLICATION

This is the main Flask entry point for the project:
"Future Interest Analysis in Twitter Users Using Data Mining"

It wires together:
    - preprocessing/  (text cleaning)
    - modules/         (interest prediction, sentiment, trends, recommendations)
    - database/        (SQLite persistence)
    - templates/        (HTML pages)
    - static/           (CSS / JS)

Run with:
    python app.py

Then open the printed localhost URL in your browser.
"""

import os
import json
import traceback
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import joblib

from preprocessing.clean_text import clean_text
from modules.interest_prediction import analyze_user_tweets, monthly_trend, predict_category
from modules.sentiment_analysis import analyze_user_sentiment
from modules.recommendation import get_recommendations
from modules.trend_analysis import extract_top_keywords, extract_hashtags, extract_hashtags_from_column
from database import database

# ---------------------------------------------------------------------
# Flask app configuration
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "dataset")
ALLOWED_EXTENSIONS = {"csv"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB upload limit (MODULE 23: SECURITY)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "college-mini-project-secret-key"  # fine for an academic demo

REQUIRED_COLUMNS = ["user_id", "username", "tweet", "category"]
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def allowed_file(filename):
    """MODULE 23: SECURITY - only allow .csv uploads."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def model_is_trained():
    """Checks whether a trained model + vectorizer already exist on disk."""
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    return os.path.exists(model_path) and os.path.exists(vectorizer_path)


def load_model_and_vectorizer():
    """Loads the trained model and TF-IDF vectorizer from disk."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    return model, vectorizer


def load_model_metrics():
    """Loads the saved model comparison metrics (accuracy/precision/etc)."""
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


# Initialize the database tables on startup.
database.init_db()


# ---------------------------------------------------------------------
# ERROR HANDLING (MODULE 22)
# ---------------------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html", error_title="Page Not Found",
                            error_message="The page you are looking for does not exist."), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("error.html", error_title="Something Went Wrong",
                            error_message="An unexpected error occurred. Please try again."), 500


# ---------------------------------------------------------------------
# ROUTE: Home Page
# ---------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------------------
# ROUTE: Dataset Upload (MODULE 1: DATA COLLECTION)
# ---------------------------------------------------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "dataset_file" not in request.files:
            flash("No file part in the request. Please choose a CSV file.", "danger")
            return redirect(request.url)

        file = request.files["dataset_file"]

        if file.filename == "":
            flash("No file selected. Please choose a CSV file to upload.", "danger")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Invalid file type. Only .csv files are allowed.", "danger")
            return redirect(request.url)

        try:
            filename = secure_filename("twitter_dataset.csv")
            final_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

            # Save to a TEMPORARY path first and validate before overwriting
            # the existing good dataset. This prevents a bad/invalid upload
            # from corrupting a previously working dataset.
            temp_path = os.path.join(app.config["UPLOAD_FOLDER"], "_upload_temp.csv")
            file.save(temp_path)

            df = pd.read_csv(temp_path)
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                os.remove(temp_path)
                flash(f"Uploaded CSV is missing required column(s): {missing_cols}. "
                      f"Required columns: {REQUIRED_COLUMNS}. The previous dataset was kept unchanged.", "danger")
                return redirect(request.url)

            if df.empty:
                os.remove(temp_path)
                flash("Uploaded CSV is empty. Please upload a dataset with data rows. "
                      "The previous dataset was kept unchanged.", "danger")
                return redirect(request.url)

            # Validation passed -- now it's safe to replace the real dataset file.
            #
            # WINDOWS NOTE: Antivirus / Windows Search Indexer sometimes takes
            # a very brief lock on a just-written file, causing an
            # intermittent "WinError 5: Access is denied" on the very first
            # rename attempt. Retrying a few times with a short pause resolves
            # this without any user action required.
            import time
            last_error = None
            replaced = False
            for attempt in range(5):
                try:
                    os.replace(temp_path, final_path)
                    replaced = True
                    break
                except PermissionError as e:
                    last_error = e
                    time.sleep(0.4)

            if not replaced:
                # Final fallback: copy the bytes over instead of renaming,
                # then remove the temp file. This avoids the Windows rename
                # lock entirely.
                try:
                    import shutil
                    shutil.copyfile(temp_path, final_path)
                    os.remove(temp_path)
                except Exception:
                    flash(f"Could not save the uploaded dataset because another "
                          f"program has the file open (Windows error: {last_error}). "
                          f"Please close Excel/VS Code preview tabs for "
                          f"twitter_dataset.csv and try again.", "danger")
                    return redirect(request.url)

            # Store the uploaded tweets in the SQLite database as well.
            database.clear_tweets()
            database.insert_tweets_dataframe(df)

            flash(f"Dataset uploaded successfully! {len(df)} rows loaded. "
                  f"Now click 'Train Model' below to train the classifiers.", "success")
            return redirect(url_for("upload"))

        except pd.errors.EmptyDataError:
            flash("The uploaded file is empty or not a valid CSV. The previous dataset was kept unchanged.", "danger")
            return redirect(request.url)
        except Exception as e:
            flash(f"Error processing uploaded file: {str(e)}", "danger")
            return redirect(request.url)

    dataset_exists = os.path.exists(os.path.join(UPLOAD_FOLDER, "twitter_dataset.csv"))
    row_count = None
    if dataset_exists:
        try:
            df = pd.read_csv(os.path.join(UPLOAD_FOLDER, "twitter_dataset.csv"))
            row_count = len(df)
        except Exception:
            row_count = None

    return render_template("upload.html", dataset_exists=dataset_exists,
                            row_count=row_count, model_trained=model_is_trained())


# ---------------------------------------------------------------------
# ROUTE: Train Model (triggers train_model.py logic)
# ---------------------------------------------------------------------
@app.route("/train", methods=["POST"])
def train():
    try:
        import train_model
        train_model.main()
        flash("Model trained successfully! Check the Model Performance page for results.", "success")
    except FileNotFoundError as e:
        flash(str(e), "danger")
    except ValueError as e:
        flash(str(e), "danger")
    except Exception as e:
        flash(f"Training failed: {str(e)}", "danger")
    return redirect(url_for("upload"))


# ---------------------------------------------------------------------
# ROUTE: Model Performance (MODULE 7: MODEL EVALUATION)
# ---------------------------------------------------------------------
@app.route("/model-performance")
def model_performance():
    metrics = load_model_metrics()
    if metrics is None:
        flash("No trained model found yet. Please upload a dataset and train the model first.", "warning")
        return redirect(url_for("upload"))
    return render_template("model_performance.html", metrics=metrics)


# ---------------------------------------------------------------------
# ROUTE: Prediction Dashboard / User Analysis (MODULES 8-14, 18-19)
# ---------------------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    if not model_is_trained():
        flash("No trained model found yet. Please upload a dataset and train the model first.", "warning")
        return redirect(url_for("upload"))

    users = database.get_distinct_users()
    df_all = database.get_all_tweets_df()

    overview = None
    if not df_all.empty:
        # Aggregate stats across the WHOLE uploaded dataset (ground-truth
        # 'category' labels from the CSV, not model predictions) -- gives a
        # quick birds-eye view of the dataset before drilling into one user.
        category_counts = df_all["category"].value_counts()
        total_tweets = len(df_all)

        cleaned_tweets = [clean_text(t) for t in df_all["tweet"].tolist()]
        top_keywords = extract_top_keywords(cleaned_tweets, top_n=8)

        if "hashtags" in df_all.columns and df_all["hashtags"].notna().any():
            top_hashtags = extract_hashtags_from_column(df_all["hashtags"].tolist())[:8]
        else:
            top_hashtags = extract_hashtags(df_all["tweet"].tolist())[:8]

        sentiment_overview = analyze_user_sentiment(df_all["tweet"].tolist())

        overview = {
            "total_tweets": total_tweets,
            "total_users": len(users),
            "category_labels": list(category_counts.index),
            "category_values": [int(v) for v in category_counts.values],
            "top_keywords": top_keywords,
            "top_hashtags": top_hashtags,
            "sentiment": sentiment_overview,
        }

    return render_template("dashboard.html", users=users, overview=overview)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not model_is_trained():
        flash("No trained model found yet. Please upload a dataset and train the model first.", "warning")
        return redirect(url_for("upload"))

    users = database.get_distinct_users()

    if request.method == "POST":
        user_identifier = request.form.get("username", "").strip()

        if not user_identifier:
            flash("Please select or enter a username / user ID.", "danger")
            return render_template("prediction.html", users=users, result=None)

        try:
            df_user = database.get_user_tweets_df(user_identifier)

            if df_user.empty:
                flash(f"No tweets found for user '{user_identifier}'. "
                      f"Please check the username/user ID and try again.", "danger")
                return render_template("prediction.html", users=users, result=None)

            model, vectorizer = load_model_and_vectorizer()

            # MODULE 9: FUTURE INTEREST PREDICTION
            analysis = analyze_user_tweets(df_user, model, vectorizer, clean_text)

            # MODULE 10: INTEREST TREND ANALYSIS (monthly line-chart data)
            df_with_predictions = pd.DataFrame(analysis["detailed_predictions"])
            trend_chart_data = monthly_trend(df_with_predictions) if not df_with_predictions.empty else {}

            # MODULE 11: SENTIMENT ANALYSIS
            sentiment_result = analyze_user_sentiment(df_user["tweet"].tolist())

            # MODULE 12: KEYWORD ANALYSIS
            cleaned_tweets = [clean_text(t) for t in df_user["tweet"].tolist()]
            top_keywords = extract_top_keywords(cleaned_tweets, top_n=10)

            # MODULE 13: HASHTAG ANALYSIS
            if "hashtags" in df_user.columns and df_user["hashtags"].notna().any():
                top_hashtags = extract_hashtags_from_column(df_user["hashtags"].tolist())
            else:
                top_hashtags = extract_hashtags(df_user["tweet"].tolist())

            # MODULE 14: RECOMMENDATION ENGINE
            primary_interest = analysis["primary_future_interest"]
            recommendations = get_recommendations(primary_interest) if primary_interest else []

            # MODULE 20: SAVE TO DATABASE (prediction history)
            username_display = df_user["username"].iloc[0] if "username" in df_user.columns else user_identifier
            database.save_prediction(
                user_id=user_identifier,
                username=username_display,
                predicted_interest=primary_interest,
                emerging_interest=analysis["potential_emerging_interest"],
                confidence=analysis["average_confidence"],
            )
            database.save_sentiment_result(
                user_id=user_identifier,
                breakdown=sentiment_result["breakdown"],
                dominant_sentiment=sentiment_result["dominant_sentiment"],
            )
            database.save_recommendation(user_identifier, primary_interest, recommendations)

            result = {
                "user_identifier": user_identifier,
                "username": username_display,
                "analysis": analysis,
                "trend_chart_data": trend_chart_data,
                "sentiment": sentiment_result,
                "top_keywords": top_keywords,
                "top_hashtags": top_hashtags,
                "recommendations": recommendations,
            }

            return render_template("prediction.html", users=users, result=result)

        except Exception as e:
            traceback.print_exc()
            flash(f"Error analyzing user: {str(e)}", "danger")
            return render_template("prediction.html", users=users, result=None)

    return render_template("prediction.html", users=users, result=None)


# ---------------------------------------------------------------------
# ROUTE: Prediction History (MODULE 20: DATABASE)
# ---------------------------------------------------------------------
@app.route("/history")
def history():
    predictions = database.get_prediction_history(limit=100)
    return render_template("history.html", predictions=predictions)


# ---------------------------------------------------------------------
# ROUTE: About Project
# ---------------------------------------------------------------------
@app.route("/about")
def about():
    return render_template("about.html")


# ---------------------------------------------------------------------
# API-style endpoint used by dashboard.html JavaScript for live single
# tweet classification demo (MODULE 8: INTEREST CLASSIFICATION)
# ---------------------------------------------------------------------
@app.route("/api/classify-tweet", methods=["POST"])
def api_classify_tweet():
    if not model_is_trained():
        return jsonify({"error": "Model not trained yet."}), 400

    data = request.get_json(silent=True) or {}
    tweet_text = data.get("tweet", "").strip()

    if not tweet_text:
        return jsonify({"error": "Please enter some tweet text."}), 400

    try:
        model, vectorizer = load_model_and_vectorizer()
        result = predict_category(tweet_text, model, vectorizer, clean_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Future Interest Analysis in Twitter Users Using Data Mining")
    print("Starting Flask development server...")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)