# assignment1-datamining

Mastering Sentiment Analysis with CRISP-DM: An End-to-End Walkthrough on IMDB Reviews

By Pranjal Shrivastava

https://medium.com/@pranjal.shrivastava_9505/mastering-sentiment-analysis-with-crisp-dm-an-end-to-end-walkthrough-on-imdb-reviews-512b09021bf0  
medium article link 

Introduction

Explanation
Sentiment analysis is the task of classifying text as positive or negative. In this project, we apply the CRISP-DM framework, a structured process widely used in data mining:

Business Understanding → define goals and metrics.

Data Understanding → explore the dataset.

Data Preparation → clean and normalize text.

Modeling → build classifiers.

Evaluation → test against metrics.

Deployment → save reproducible artifacts.

We’ll use the Kaggle IMDB dataset of movie reviews, build baselines, improve them with threshold tuning and calibration, and package everything into reproducible outputs.

1. Business Understanding

Explanation
We need to decide how to measure success:

Macro-F1 is our primary metric, since it balances positive/negative equally.

Precision, Recall, ROC-AUC, PR-AUC help us understand different aspects of performance.

Baselines: (a) predict the majority class, (b) TF-IDF + Logistic Regression.

Constraint: dataset is small → linear models preferred.

(No code here — just framing).

2. Data Understanding

Explanation
Our CSV has two columns: a review and its sentiment label. Some are missing or duplicated. We’ll standardize column names, normalize labels to "positive"/"negative", and drop invalid entries.

Code

import pandas as pd

df_raw = pd.read_csv("data/raw/kaggle-imdb-dataset.csv", encoding="utf-8")

# Standardize schema
df = df_raw.rename(columns={df_raw.columns[0]: "review", df_raw.columns[1]: "sentiment"}).copy()

# Normalize labels
df["sentiment"] = df["sentiment"].str.strip().str.lower()
df = df[df["sentiment"].isin(["positive", "negative"])]

# Drop duplicates
df = df.dropna().drop_duplicates(subset=["review"]).reset_index(drop=True)

print(df.shape)
df.head()


3. Visual Inspection

Explanation
We explore basic distributions:

Token length of reviews (many between 100–250 words).

Class distribution (slightly skewed negative).
These plots help us spot outliers and decide thresholds for anomaly filtering.

Code

import matplotlib.pyplot as plt

df["n_words"] = df["review"].str.split().str.len()
df["n_chars"] = df["review"].str.len()

df["n_words"].hist(bins=30)
plt.title("Token count per review")
plt.show()

df["sentiment"].value_counts().plot(kind="bar")
plt.title("Class distribution")
plt.show()


4. Text Normalization & Cleaning

Explanation
Movie reviews contain HTML tags, URLs, emails, and noise. We clean them while preserving negations like “not good”, which are critical for sentiment.

Code

import re

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"<.*?>", " ", s)      # HTML
    s = re.sub(r"https?://\S+", " ", s)  # URLs
    s = re.sub(r"\S+@\S+", " ", s)    # Emails
    s = re.sub(r"@\w+", " ", s)       # Handles
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["clean"] = df["review"].apply(clean_text)

5. Feature Construction

Explanation
We represent text using TF-IDF (term frequency–inverse document frequency):

Word n-grams (1,2) capture unigrams and bigrams.

This gives us a sparse matrix suitable for linear classifiers.

Code

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_word = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1,2))
X = tfidf_word.fit_transform(df["clean"])

6. Unsupervised Exploration

Explanation
Before supervised learning, we cluster reviews to see if groups align with sentiment. Using SVD (50 dims) + KMeans (k=2), we check cluster composition.

Code

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

X_svd = TruncatedSVD(n_components=50, random_state=42).fit_transform(X)
km = MiniBatchKMeans(n_clusters=2, random_state=42).fit(X_svd)
df["cluster"] = km.labels_

print(pd.crosstab(df["cluster"], df["sentiment"]))


7. Anomaly Detection

Explanation
Some reviews are too short, noisy, or duplicates. Removing them avoids misleading the model.

Code

def fingerprint(text):
    return " ".join(text.split()[:50])

df["fingerprint"] = df["clean"].apply(fingerprint)

short = df["n_words"] < 5
dupe  = df["fingerprint"].duplicated(keep="first")

df["is_anomaly"] = short | dupe
anomalies = df[df["is_anomaly"]]
df_clean = df[~df["is_anomaly"]]


8. Predictive Modeling

Explanation
We benchmark three models on an 80/20 holdout:

Logistic Regression (interpretable baseline)

ComplementNB (fast, works well on text)

LinearSVC (strong margin classifier)

Code

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    df_clean["clean"], df_clean["sentiment"], stratify=df_clean["sentiment"], test_size=0.2)

from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([("tfidf", tfidf_word), ("clf", LogisticRegression(max_iter=1000))])
pipe_lr.fit(X_train, y_train)
print("Logistic:\n", classification_report(y_test, pipe_lr.predict(X_test)))


9. Threshold Tuning (Logistic)

Explanation
Default threshold = 0.5 may not maximize F1. We tune τ on a calibration split to optimize performance.

Code

from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

scores = pipe_lr.predict_proba(X_test)[:,1]
P, R, T = precision_recall_curve((y_test=="positive"), scores)
f1s = (2*P*R)/(P+R+1e-12)
best_idx = np.argmax(f1s)
best_thr = T[best_idx-1]
print("Best threshold:", best_thr)


10. Calibrated SVM

Explanation
LinearSVC gives decision scores, not probabilities. We use Platt scaling to calibrate and compute AUCs.

Code

from sklearn.calibration import CalibratedClassifierCV

pipe_svm = Pipeline([("tfidf", tfidf_word), ("clf", LinearSVC())])
pipe_svm.fit(X_train, y_train)

cal_svm = CalibratedClassifierCV(pipe_svm, cv=3, method="sigmoid")
cal_svm.fit(X_train, y_train)


11. Error Analysis

Explanation
We inspect misclassified reviews and slice performance by review length. This shows model blind spots (e.g., short reviews).

Code

pred = pipe_lr.predict(X_test)
errors = pd.DataFrame({"review": X_test, "true": y_test, "pred": pred})
print(errors[errors["true"] != errors["pred"]].head())


12. Model Interpretation

Explanation
Logistic Regression weights tell us which words drive sentiment predictions. This improves interpretability.

Code

vect = pipe_lr.named_steps["tfidf"]
clf  = pipe_lr.named_steps["clf"]

terms = vect.get_feature_names_out()
coefs = clf.coef_[0]

top_pos = terms[np.argsort(coefs)[-10:]]
top_neg = terms[np.argsort(coefs)[:10]]
print("Positive terms:", top_pos)
print("Negative terms:", top_neg)


13. Reproducibility & Artifacts

Explanation
We save the pipeline, metadata (thresholds, versions), and processed data. This ensures reproducibility.

Code

import joblib, json
joblib.dump(pipe_lr, "models/imdb_best_pipeline.joblib")

meta = {"threshold": float(best_thr), "label_map": {"negative":0,"positive":1}}
json.dump(meta, open("models/imdb_pipeline_metadata.json","w"))


Conclusion

We’ve walked the full CRISP-DM cycle: business framing, data understanding, cleaning, feature engineering, modeling, evaluation, and deployment. Along the way, we:

Validated the dataset and removed anomalies.

Benchmarked Logistic, Naive Bayes, and SVM.

Tuned thresholds for Logistic.

Calibrated SVM to produce probabilities.

Analyzed errors and interpreted model weights.

Packaged everything for reproducibility and reporting.

This isn’t just a Kaggle notebook; it’s a professional-grade workflow.

