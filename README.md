# Future Interest Analysis in Twitter Users Using Data Mining

A college-level Data Mining / Machine Learning mini-project that analyzes a
Twitter/X user's historical tweets and predicts their **future interests**
using Natural Language Processing (NLP) and Machine Learning classification.

---

## 1. Project Description

Social media users constantly reveal their interests through what they post.
This project mines that behavioural signal: it cleans and classifies a
user's historical tweets into interest categories (Technology, Sports,
Entertainment, etc.), studies how those categories shift over time, and
produces a clearly-labeled **prediction** of the user's likely future
interest — along with sentiment, keyword, and hashtag analysis, and simple
topic recommendations.

> **Important academic honesty note:** The bundled dataset
> (`dataset/twitter_dataset.csv`) is **synthetic sample/demo data** generated
> for this project, not real data scraped from Twitter/X. Real-time
> Twitter/X API integration is **not implemented** — it is listed under
> Future Enhancements (Section 9).

---

## 2. Objectives

1. Accept historical tweet data from a CSV dataset.
2. Clean and preprocess tweet text (NLP pipeline).
3. Extract features using TF-IDF.
4. Train and compare Logistic Regression, Naïve Bayes, and SVM.
5. Automatically select the best-performing model using real metrics.
6. Classify tweets into 10 interest categories.
7. Aggregate a user's tweet history into a future-interest prediction.
8. Detect increasing/decreasing/emerging interests over time.
9. Perform sentiment analysis (Positive / Negative / Neutral).
10. Extract top keywords and hashtags.
11. Generate simple rule-based recommendations.
12. Present everything through an interactive Flask + Chart.js dashboard.
13. Store all predictions in SQLite for historical review.

---

## 3. Features

- CSV dataset upload with validation (column checks, empty-file checks, size limit)
- Full NLP preprocessing pipeline: lowercasing, URL/mention/emoji removal,
  punctuation/number stripping, stop-word removal, tokenization, lemmatization
- TF-IDF feature extraction (unigrams + bigrams)
- Three ML classifiers trained and compared on real metrics (Accuracy,
  Precision, Recall, F1-score, Confusion Matrix) — **no hard-coded numbers**
- Automatic best-model selection
- Future Interest Prediction (primary interest + potential emerging interest)
- Interest Trend Analysis (older vs. recent activity, monthly trend chart)
- Sentiment Analysis (VADER, via NLTK)
- Keyword & Hashtag frequency analysis
- Rule-based Recommendation Engine
- SQLite-backed Prediction History
- Responsive Bootstrap 5 + Chart.js dashboard UI
- Friendly error handling (no raw Python tracebacks shown to the user)

---

## 4. Technology Stack

| Layer          | Technology                                  |
|----------------|----------------------------------------------|
| Frontend       | HTML5, CSS3, Bootstrap 5, JavaScript, Chart.js |
| Backend        | Python 3.10+, Flask                          |
| Machine Learning | Scikit-learn (Logistic Regression, Naïve Bayes, SVM) |
| NLP            | NLTK (tokenization, lemmatization, stopwords, VADER sentiment) |
| Data Processing| Pandas, NumPy                                |
| Visualization  | Chart.js (client-side), Matplotlib (optional/offline use) |
| Database       | SQLite                                       |
| Model Storage  | Joblib / Pickle                              |

---

## 5. System Requirements

- Python 3.10 or higher
- pip
- ~200 MB free disk space (for Python packages + NLTK data)
- A modern web browser (Chrome, Firefox, Edge)
- Internet access on first run (to download NLTK corpora and CDN assets for Bootstrap/Chart.js/fonts)

---

## 6. Installation

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

NLTK resources (`punkt`, `stopwords`, `wordnet`, `vader_lexicon`, etc.) are
downloaded **automatically the first time** you run the app or the training
script — no manual step needed, as long as you have an internet connection.

---

## 7. Dataset Format

Required CSV columns:

| Column      | Description                                  |
|-------------|-----------------------------------------------|
| `user_id`   | Unique identifier for the user                |
| `username`  | Twitter/X handle                              |
| `tweet`     | Raw tweet text                                |
| `category`  | Ground-truth interest label (used for training) |

Optional columns: `date`, `hashtags`, `likes`, `retweets`.

A ready-to-use **sample/demo dataset** (300 synthetic tweets, 20 users, 10
balanced categories) is already included at `dataset/twitter_dataset.csv`.

Interest categories used: **Technology, Sports, Entertainment, Politics,
Business, Education, Health, Travel, Gaming, Finance** (easy to extend —
just add more rows with a new category label and retrain).

---

## 8. Model Training

Train (or retrain) the classifiers directly from the command line:

```bash
python train_model.py
```

This will:
1. Load and validate `dataset/twitter_dataset.csv`
2. Clean & deduplicate tweets
3. Extract TF-IDF features
4. Split into train/test (80/20, stratified)
5. Train Logistic Regression, Naïve Bayes, and SVM
6. Evaluate all three with real metrics
7. Save the best model to `models/best_model.pkl`, the vectorizer to
   `models/tfidf_vectorizer.pkl`, and metrics to `models/model_metrics.json`

You can also trigger training from the web UI: **Upload Dataset page → Train Model** button.

---

## 9. Running the Application

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

Typical demo flow:
1. **Home** — project overview
2. **Upload Dataset** — use the included sample CSV, or upload your own, then click **Train Model**
3. **Dashboard** — see aggregate stats/charts across the whole dataset, and try the live tweet classifier
4. **Analyze User** — pick a username (e.g. `rahul_k`, `gamer_karthik`) and click **Analyze User**
5. **Model Performance** — compare Logistic Regression / Naïve Bayes / SVM metrics and confusion matrix
6. **Prediction History** — review every past analysis, stored in SQLite
7. **About** — pipeline explanation and ethical notes

---

## 10. Testing Instructions

Quick manual test checklist for a project demo/viva:

1. Start the app (`python app.py`) and confirm the Home page loads.
2. Go to **Upload Dataset** — confirm the sample dataset shows a row count and "Train Model" works.
3. Go to **Model Performance** — confirm three models are compared with real (non-identical, dataset-driven) numbers.
4. Go to **Analyze User**, type `rahul_k`, click Analyze — confirm cards, charts, keywords, hashtags, and recommendations render.
5. Go to **Prediction History** — confirm the analysis you just ran appears in the table.
6. Try uploading an invalid CSV (missing `category` column) — confirm a friendly error appears **and the existing good dataset is not overwritten**.
7. Try analyzing a username that doesn't exist — confirm a friendly "No tweets found" message appears instead of a crash.

---

## 11. Project Structure

```
Future-Interest-Analysis/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── twitter_dataset.csv        (synthetic sample/demo data)
│
├── models/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_metrics.json
│
├── preprocessing/
│   ├── __init__.py
│   ├── clean_text.py              (cleaning, tokenization, lemmatization)
│   └── feature_extraction.py      (TF-IDF)
│
├── modules/
│   ├── __init__.py
│   ├── interest_prediction.py     (classification + future interest logic)
│   ├── sentiment_analysis.py      (VADER sentiment)
│   ├── recommendation.py          (rule-based recommendations)
│   └── trend_analysis.py          (keyword & hashtag extraction)
│
├── database/
│   ├── __init__.py
│   ├── database.py
│   └── database.db                (created on first run)
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── dashboard.html
│   ├── prediction.html
│   ├── history.html
│   ├── model_performance.html
│   ├── about.html
│   └── error.html
│
└── static/
    ├── css/style.css
    └── js/script.js
```

---

## 12. Screenshots

*(Add screenshots of the Home page, Dashboard, User Analysis results, and
Model Performance page here after running the app locally — e.g.
`static/images/home.png`, `static/images/dashboard.png`, etc.)*

---

## 13. Sample Output

**Model comparison (from an actual training run on the bundled dataset):**

```
Logistic Regression: Accuracy 66.67% | Precision 73.28% | Recall 66.67% | F1 66.20%
Naïve Bayes:         Accuracy 74.07% | Precision 78.84% | Recall 74.07% | F1 73.46%  <-- Best Model
SVM:                 Accuracy 70.37% | Precision 71.77% | Recall 70.37% | F1 69.43%
```

**Example future interest prediction (user: `rahul_k`):**

```
Current Interests:  Technology 46% | Education 31% | ...
Primary Future Interest:      Technology
Potential Emerging Interest:  Education
Average Confidence:           ~55-70% (varies per run/dataset)
Disclaimer: This is a prediction based on historical tweet patterns —
            not a guaranteed future behaviour.
```

---

## 14. Ethical & Responsible Use Notes

- Predictions are **probabilistic estimates based on historical data**, not guarantees.
- The bundled dataset is **synthetic/demo data**, clearly not real Twitter/X data.
- User privacy should be protected; avoid collecting sensitive personal data unnecessarily.
- Any real Twitter/X API integration must follow Twitter/X's platform policies and applicable data-protection laws.
- This system must **not** be used for harmful profiling or discriminatory decision-making.

---

## 15. Future Enhancements

1. Real-time Twitter/X API integration
2. BERT / RoBERTa-based NLP models
3. Deep learning classifiers (LSTM, Transformers)
4. Multilingual support
5. Emotion detection (beyond positive/negative/neutral)
6. Real-time trend prediction
7. Advanced (collaborative/content-based) recommendation algorithms
8. Mobile application
9. Cloud deployment
10. More sophisticated temporal/time-series prediction models

---

## 16. Viva Questions & Answers

**Q1. What is the objective of this project?**
A: To analyze a Twitter/X user's historical tweets and predict their future
interests using NLP for text processing and Machine Learning for
classification, then present the results through a web dashboard.

**Q2. Why do you need to clean the text before using it?**
A: Raw tweets contain noise — URLs, @mentions, emojis, punctuation, stop
words — that don't help distinguish between interest categories. Cleaning
removes this noise so the model can focus on words that actually carry
meaning.

**Q3. What is tokenization and why is it needed?**
A: Tokenization splits a sentence into individual words (tokens), e.g.
`"I love AI"` → `["I", "love", "AI"]`. Machine Learning models and further
NLP steps (like stop-word removal and lemmatization) operate on individual
tokens, not whole sentences.

**Q4. What is lemmatization, and how is it different from stemming?**
A: Lemmatization reduces a word to its dictionary base form using
vocabulary and grammar rules (e.g. "running" → "run", "better" → "good").
Stemming just chops off word endings using simple rules and can produce
non-words (e.g. "studies" → "studi"). Lemmatization is more accurate but
slightly slower — a reasonable trade-off for this project's dataset size.

**Q5. What is TF-IDF and why did you choose it over simple word counts?**
A: TF-IDF (Term Frequency–Inverse Document Frequency) scores a word high if
it appears often in one tweet but rarely across all tweets, meaning it's
distinctive for that tweet's topic. Common words like "the" or "is" appear
in almost every tweet, so they get a low score automatically — something
plain word counting can't do on its own.

**Q6. Why did you compare three different models instead of using just one?**
A: Different algorithms make different assumptions about the data.
Comparing Logistic Regression (a strong general-purpose linear classifier),
Naïve Bayes (traditionally strong for text/word-frequency data), and SVM
(effective in high-dimensional TF-IDF spaces) lets us empirically pick the
best fit for this specific dataset instead of guessing.

**Q7. How do you decide which model is "best"?**
A: We use the F1-score on a held-out test set (20% of the data, unseen
during training) because it balances precision and recall — useful when
some interest categories have fewer examples than others. The model with
the highest F1-score is automatically selected and saved.

**Q8. What do Accuracy, Precision, Recall, and F1-score mean here?**
A: Accuracy = overall percent correct. Precision = of the tweets predicted
as category X, how many actually were X. Recall = of all actual category X
tweets, how many did we correctly find. F1-score = harmonic mean of
precision and recall, giving a single balanced number.

**Q9. How is "future interest" different from just classifying one tweet?**
A: Classifying a single tweet only tells you today's topic. Future interest
prediction aggregates **all** of a user's historical tweets, calculates the
overall frequency of each predicted category, and compares older tweets to
recent tweets to see which categories are growing. The category with the
highest overall frequency becomes the "Primary Future Interest," and a
category growing fastest recently becomes the "Potential Emerging
Interest."

**Q10. Is the future interest prediction guaranteed to be correct?**
A: No. It's explicitly labeled as a probabilistic estimate based on past
behavioral patterns, not a guarantee — this is stated directly on the
dashboard and in the About page.

**Q11. What is sentiment analysis, and how is it different from interest
classification?**
A: Sentiment analysis (using NLTK's VADER) determines whether a tweet's
*tone* is Positive, Negative, or Neutral. Interest classification determines
the tweet's *topic* (Technology, Sports, etc.). They are separate,
complementary signals — a Technology tweet can be positive or negative
independent of it being about Technology.

**Q12. Why did you choose VADER for sentiment instead of training your own
sentiment classifier?**
A: VADER is specifically tuned for short, informal social-media text and
doesn't require a separately labeled training set, which keeps the project
scope reasonable while still giving a genuinely useful sentiment signal.

**Q13. How does the recommendation engine work?**
A: It's a simple rule-based lookup: once a user's primary predicted
interest category is known, a fixed list of related topics for that
category is returned (e.g. Technology → AI, Python, Cloud Computing,
Cybersecurity). This is transparent and easy to extend with more advanced
algorithms later.

**Q14. Why SQLite instead of MySQL/PostgreSQL?**
A: SQLite is a lightweight, file-based database that requires no separate
server setup — ideal for a self-contained academic prototype that should
run anywhere Python runs, while still supporting real SQL queries and
structured tables.

**Q15. What tables are in your database, and why?**
A: `users` (registered usernames), `tweets` (uploaded tweet data),
`predictions` (future-interest prediction history), `sentiment_results`
(sentiment history), and `recommendations` (recommendation history). This
separation keeps each concern independently queryable and auditable.

**Q16. How do you handle a user who doesn't exist in the dataset?**
A: The system checks if any tweets match the given username/user ID before
running analysis. If none are found, it shows a friendly message instead
of crashing.

**Q17. What happens if someone uploads an invalid CSV?**
A: The file is validated (correct extension, required columns present,
non-empty) *before* it replaces the existing dataset. If validation fails,
a clear error message is shown and the previously working dataset is left
untouched.

**Q18. Is this using real Twitter/X data?**
A: No — the bundled dataset is synthetic sample data created for
demonstration purposes only, and this is clearly disclosed in the README,
the About page, and the code comments. Real-time API integration is listed
as a future enhancement, not something currently implemented.

**Q19. How would you extend this project to more interest categories?**
A: Add tweets labeled with the new category to the training CSV and
retrain (`python train_model.py`). Because the categories are learned from
the data rather than hard-coded into the model logic, the classifier
automatically adapts to whatever categories are present in the labeled
dataset. (The rule-based recommendation engine's topic list would need a
small manual addition for a brand-new category.)

**Q20. What are the main limitations of this project?**
A: Small demo dataset size, English-only, TF-IDF misses word context and
order more than modern embedding-based methods would, historical patterns
don't strictly guarantee future behavior, and it currently has no real-time
Twitter/X API connection — all disclosed under Future Enhancements.

---

## 17. Disclaimer

This is an academic prototype built for a college Data Mining / Machine
Learning mini-project. It is not intended for production deployment,
commercial profiling, or real-world behavioral targeting without further
review of data privacy, security, and platform-policy compliance.
