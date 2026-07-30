"""
Resume Category Classifier — simple, self-contained Streamlit app.

Only needs: streamlit, pandas, numpy, scikit-learn
(no nltk, no altair, no external image files — everything is in this
one file, so nothing can go missing.)

Run with:
    streamlit run app.py

Put "Resume.csv" in the same folder, or upload it from the sidebar.
"""

import re
import string
import os

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# --------------------------------------------------------------------------
# Page setup + colourful styling
# --------------------------------------------------------------------------
st.set_page_config(page_title="Resume Category Classifier", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7f7ff 0%, #f2fbfa 60%, #fffaf0 100%); }
    .hero-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #6C63FF, #4DABF7, #38D9A9, #FFC857);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .card {
        border-radius: 16px; padding: 1.1rem 1.4rem; margin-bottom: 0.8rem;
        box-shadow: 0 6px 16px rgba(60,60,120,0.08);
    }
    .card-purple { background: linear-gradient(135deg, #efe9ff, #f6f3ff); border-left: 6px solid #6C63FF; }
    .card-blue   { background: linear-gradient(135deg, #e8f4ff, #f2f9ff); border-left: 6px solid #4DABF7; }
    .card-teal   { background: linear-gradient(135deg, #e4faf3, #eefdf8); border-left: 6px solid #38D9A9; }
    .card-gold   { background: linear-gradient(135deg, #fff6e0, #fffaee); border-left: 6px solid #FFC857; }
    .result-box {
        border-radius: 18px; padding: 1.6rem;
        background: linear-gradient(135deg, #6C63FF 0%, #4DABF7 45%, #38D9A9 100%);
        color: white; text-align: center;
    }
    .result-box h1 { color: white; margin: 0.3rem 0; font-size: 1.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-title">Resume Category Classifier</p>', unsafe_allow_html=True)
st.write("Paste in a resume and a simple text-classification model will guess its career field.")
st.write("")

# --------------------------------------------------------------------------
# Lightweight text cleaning (no nltk needed)
# --------------------------------------------------------------------------
BASIC_STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "i'm": "i am", "it's": "it is", "i've": "i have", "didn't": "did not",
    "doesn't": "does not", "isn't": "is not", "wasn't": "was not",
    "you're": "you are", "they're": "they are", "we're": "we are",
}


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", "", str(text))
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [CONTRACTIONS.get(w.lower(), w) for w in words]
    words = [w for w in words if w.lower() not in BASIC_STOPWORDS]
    return " ".join(words)


# --------------------------------------------------------------------------
# Data loading — flexible about column names/layout
# --------------------------------------------------------------------------
def load_dataframe(file_like_or_path):
    df = pd.read_csv(file_like_or_path)

    # If it came in as one big unsplit column (like the raw notebook CSV),
    # split it out.
    if df.shape[1] == 1:
        df = df.iloc[:, 0].str.split(",", n=4, expand=True)
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = [str(c).strip().lower() for c in df.columns]

    # try to find the text column and category column by common names
    text_col = next((c for c in df.columns if c in ("resume_text", "resume_str", "text", "resume")), None)
    cat_col = next((c for c in df.columns if c in ("category", "label", "resume_category")), None)

    if text_col is None or cat_col is None:
        raise ValueError(
            f"Couldn't find a resume-text column and a category column. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[[text_col, cat_col]].rename(columns={text_col: "resume_text", cat_col: "category"})
    df = df.dropna(subset=["resume_text", "category"])
    return df


# --------------------------------------------------------------------------
# Train model (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def train_pipeline(data_source):
    df = load_dataframe(data_source)
    df["clean_text"] = df["resume_text"].astype(str).apply(clean_text)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["category"])

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df["clean_text"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    category_counts = df["category"].value_counts()

    return {
        "model": model,
        "tfidf": tfidf,
        "encoder": encoder,
        "accuracy": accuracy,
        "category_counts": category_counts,
        "n_samples": len(df),
    }


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "Resume.csv")

with st.sidebar:
    st.markdown("### Dataset")
    uploaded_csv = st.file_uploader("Upload Resume.csv", type=["csv"])
    st.caption("If nothing is uploaded, the app looks for Resume.csv in its own folder.")

data_source = uploaded_csv if uploaded_csv is not None else (
    DEFAULT_DATA_PATH if os.path.exists(DEFAULT_DATA_PATH) else None
)

if data_source is None:
    st.markdown(
        '<div class="card card-gold"><b>No dataset found.</b><br>'
        "Add Resume.csv next to app.py, or upload it from the sidebar.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

try:
    pipeline = train_pipeline(data_source)
except Exception as e:
    st.error(f"Couldn't train the model: {e}")
    st.stop()

# --------------------------------------------------------------------------
# Overview cards
# --------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="card card-purple"><b>Resumes used</b><br><span style="font-size:1.5rem;">{pipeline["n_samples"]}</span></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="card card-blue"><b>Categories</b><br><span style="font-size:1.5rem;">{len(pipeline["encoder"].classes_)}</span></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="card card-teal"><b>Test accuracy</b><br><span style="font-size:1.5rem;">{pipeline["accuracy"]*100:.1f}%</span></div>', unsafe_allow_html=True)

with st.expander("Category distribution in the dataset"):
    st.bar_chart(pipeline["category_counts"])

st.write("")

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
st.markdown("#### Try it on a resume")
resume_text = st.text_area(
    "Paste resume text here",
    height=220,
    placeholder="e.g. Skilled Python developer with experience in machine learning, "
    "SQL, data analysis, and building predictive models...",
)

if st.button("Analyze resume", type="primary"):
    if not resume_text.strip():
        st.markdown('<div class="card card-gold">Please paste some resume text first.</div>', unsafe_allow_html=True)
    else:
        cleaned = clean_text(resume_text)
        vector = pipeline["tfidf"].transform([cleaned])
        pred = pipeline["model"].predict(vector)[0]
        proba = pipeline["model"].predict_proba(vector)[0]
        label = pipeline["encoder"].inverse_transform([pred])[0]
        confidence = float(np.max(proba)) * 100

        st.markdown(
            f"""
            <div class="result-box">
                <div>Predicted category</div>
                <h1>{label}</h1>
                <div>Confidence: {confidence:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("##### Top 5 category probabilities")
        proba_series = pd.Series(proba, index=pipeline["encoder"].classes_).sort_values(ascending=False).head(5)
        st.bar_chart(proba_series)

        st.caption("A simple TF-IDF + Naive Bayes model — treat this as a first impression, not a final verdict.")
