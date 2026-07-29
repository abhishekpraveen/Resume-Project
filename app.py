import re
import string

import numpy as np
import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# ----------------------------------------------------------------------
# One-time NLTK setup
# ----------------------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
        nltk.download(pkg, quiet=True)


download_nltk_data()
STOP_WORDS = set(stopwords.words("english"))

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "I'm": "I am",
    "it's": "it is",
}


# ----------------------------------------------------------------------
# Preprocessing helpers (mirrors resume.ipynb exactly)
# ----------------------------------------------------------------------
def remove_html(text):
    return re.sub(r"<.*?>", "", str(text))


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def expand(text):
    words = text.split()
    words = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    return " ".join(words)


def remove_stopwords(text):
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in STOP_WORDS]
    return " ".join(words)


def preprocess(text):
    text = remove_html(text)
    text = remove_punctuation(text)
    text = expand(text)
    text = remove_stopwords(text)
    return text


# ----------------------------------------------------------------------
# Load data + train model (cached so it only runs once per deployment)
# ----------------------------------------------------------------------
@st.cache_resource
def load_and_train():
    # Same parsing logic as the notebook
    df = pd.read_csv("Resume.csv", header=None)
    df = df[0].str.split(",", n=4, expand=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)

    df["resume_text"] = df["resume_text"].apply(preprocess)

    encoder = LabelEncoder()
    df["category_encoded"] = encoder.fit_transform(df["category"])

    tfidf = TfidfVectorizer(max_features=3000)
    X_tfidf = tfidf.fit_transform(df["resume_text"])
    y = df["category_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, tfidf, encoder, accuracy


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Resume Category Classifier", page_icon="📄")

st.title("📄 Resume Category Classifier")
st.write(
    "Paste resume text below and the model will predict which job "
    "category it belongs to."
)

with st.spinner("Loading data and training model..."):
    model, tfidf, encoder, accuracy = load_and_train()

st.caption(f"Model test accuracy: **{accuracy:.2%}**")

resume_text = st.text_area(
    "Resume text",
    height=250,
    placeholder="Paste resume text here...",
)

if st.button("Predict Category", type="primary"):
    if not resume_text.strip():
        st.warning("Please paste some resume text first.")
    else:
        cleaned = preprocess(resume_text)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)
        predicted_category = encoder.inverse_transform(prediction)[0]

        st.success(f"**Predicted Category:** {predicted_category}")
