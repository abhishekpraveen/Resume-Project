"""
Resume Category Classifier
---------------------------
A friendly, colourful Streamlit front-end for the NLP resume-classification
pipeline: cleaning -> stopword removal -> TF-IDF -> Multinomial Naive Bayes.

Run with:
    streamlit run app.py

Make sure "Resume.csv" is in the same folder as this file (or upload your
own copy of the same dataset from the sidebar).
"""

import os
import re
import string
import base64

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Category Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS = os.path.join(os.path.dirname(__file__), "assets")


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def asset(name):
    return os.path.join(ASSETS, name)


# --------------------------------------------------------------------------
# Colourful styling (no emojis anywhere — images/icons only)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f7ff 0%, #f2fbfa 60%, #fffaf0 100%);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C63FF, #4DABF7, #38D9A9, #FFC857);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #555b6e;
        font-size: 1.05rem;
        margin-top: 0.2rem;
    }
    .card {
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 6px 18px rgba(60, 60, 120, 0.08);
    }
    .card-purple { background: linear-gradient(135deg, #efe9ff, #f6f3ff); border-left: 6px solid #6C63FF; }
    .card-blue   { background: linear-gradient(135deg, #e8f4ff, #f2f9ff); border-left: 6px solid #4DABF7; }
    .card-teal   { background: linear-gradient(135deg, #e4faf3, #eefdf8); border-left: 6px solid #38D9A9; }
    .card-gold   { background: linear-gradient(135deg, #fff6e0, #fffaee); border-left: 6px solid #FFC857; }
    .result-box {
        border-radius: 20px;
        padding: 1.8rem;
        background: linear-gradient(135deg, #6C63FF 0%, #4DABF7 45%, #38D9A9 100%);
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(80, 80, 160, 0.25);
    }
    .result-box h1 {
        color: white;
        margin: 0.3rem 0;
        font-size: 2rem;
    }
    .footer-note {
        text-align: center;
        color: #8a8fa3;
        font-size: 0.85rem;
        margin-top: 2.5rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fdfbff 0%, #f3f8ff 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Header / banner
# --------------------------------------------------------------------------
banner_path = asset("banner.png")
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)

col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image(asset("resume_icon.png"), width=90)
with col_title:
    st.markdown('<p class="hero-title">Resume Category Classifier</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Paste in a resume and let a little Naive Bayes model '
        "guess which career field it belongs to — trained entirely on your own data.</p>",
        unsafe_allow_html=True,
    )

st.write("")

# --------------------------------------------------------------------------
# NLTK data (downloaded once, cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_nltk_data():
    for pkg in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True


ensure_nltk_data()
STOP_WORDS = set(stopwords.words("english"))

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "i'm": "i am", "it's": "it is", "i've": "i have", "didn't": "did not",
    "doesn't": "does not", "isn't": "is not", "wasn't": "was not",
    "you're": "you are", "they're": "they are", "we're": "we are",
}


def clean_text(text: str) -> str:
    """Same cleaning pipeline as the original notebook: strip HTML, strip
    punctuation, expand contractions, and drop stopwords."""
    text = re.sub(r"<.*?>", "", str(text))
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [CONTRACTIONS.get(w.lower(), w) for w in words]
    text = " ".join(words)
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    tokens = [w for w in tokens if w.lower() not in STOP_WORDS]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# Data loading + model training (cached so it only happens once per session)
# --------------------------------------------------------------------------
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "Resume.csv")


def load_dataframe(file_like_or_path):
    df = pd.read_csv(file_like_or_path, header=None)
    df = df[0].str.split(",", n=4, expand=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    # normalise expected column names
    lower_cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=lower_cols)
    if "resume_text" not in df.columns or "category" not in df.columns:
        raise ValueError(
            "Couldn't find 'resume_text' and 'category' columns after parsing. "
            "Please check that Resume.csv matches the expected format."
        )
    return df


@st.cache_resource(show_spinner=True)
def train_pipeline(data_source):
    df = load_dataframe(data_source)
    df["resume_text"] = df["resume_text"].astype(str).apply(clean_text)

    encoder = LabelEncoder()
    df["category_encoded"] = encoder.fit_transform(df["category"])

    X = df["resume_text"]
    y = df["category_encoded"]

    tfidf = TfidfVectorizer(max_features=3000)
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=encoder.classes_, output_dict=True, zero_division=0
    )

    category_counts = df["category"].value_counts().reset_index()
    category_counts.columns = ["category", "count"]

    return {
        "model": model,
        "tfidf": tfidf,
        "encoder": encoder,
        "accuracy": accuracy,
        "report": report,
        "category_counts": category_counts,
        "n_samples": len(df),
    }


# --------------------------------------------------------------------------
# Sidebar — data source + about
# --------------------------------------------------------------------------
with st.sidebar:
    st.image(asset("chart_icon.png"), width=70)
    st.markdown("### About this app")
    st.write(
        "This app trains a TF-IDF + Multinomial Naive Bayes classifier on your "
        "resume dataset, then predicts the most likely job category for any "
        "resume text you provide."
    )
    st.markdown("---")
    st.markdown("### Dataset")
    uploaded_csv = st.file_uploader("Upload Resume.csv (optional)", type=["csv"])
    st.caption(
        "If you don't upload a file, the app looks for **Resume.csv** in its "
        "own folder."
    )

# --------------------------------------------------------------------------
# Train (or load cached) model
# --------------------------------------------------------------------------
data_source = None
if uploaded_csv is not None:
    data_source = uploaded_csv
elif os.path.exists(DEFAULT_DATA_PATH):
    data_source = DEFAULT_DATA_PATH

if data_source is None:
    st.markdown(
        '<div class="card card-gold">'
        "<b>No dataset found yet.</b><br>"
        "Drop <code>Resume.csv</code> next to <code>app.py</code>, or upload it "
        "from the sidebar, to get started."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

with st.spinner("Training the model on your resume dataset — just a moment..."):
    try:
        pipeline = train_pipeline(data_source)
    except Exception as e:
        st.error(f"Something went wrong while training: {e}")
        st.stop()

# --------------------------------------------------------------------------
# Overview cards
# --------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f'<div class="card card-purple"><b>Resumes used</b><br>'
        f'<span style="font-size:1.6rem;">{pipeline["n_samples"]}</span></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="card card-blue"><b>Categories learned</b><br>'
        f'<span style="font-size:1.6rem;">{len(pipeline["encoder"].classes_)}</span></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="card card-teal"><b>Test accuracy</b><br>'
        f'<span style="font-size:1.6rem;">{pipeline["accuracy"]*100:.1f}%</span></div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Category distribution chart (multi-colour bars)
# --------------------------------------------------------------------------
with st.expander("See how resumes are spread across categories", expanded=False):
    counts_df = pipeline["category_counts"]
    chart = (
        alt.Chart(counts_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("category:N", sort="-y", title="Category"),
            y=alt.Y("count:Q", title="Number of resumes"),
            color=alt.Color("category:N", legend=None, scale=alt.Scale(scheme="rainbow")),
            tooltip=["category", "count"],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

st.write("")

# --------------------------------------------------------------------------
# Prediction area
# --------------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.image(asset("upload_icon.png"), width=55)
    st.markdown("#### Try it on a resume")
    input_mode = st.radio("How would you like to provide the resume?", ["Paste text", "Upload a .txt file"], horizontal=True)

    resume_text = ""
    if input_mode == "Paste text":
        resume_text = st.text_area(
            "Paste resume text here",
            height=260,
            placeholder="e.g. Skilled Python developer with experience in machine learning, "
            "SQL, data analysis, and building predictive models...",
        )
    else:
        txt_file = st.file_uploader("Upload a .txt resume", type=["txt"], key="resume_txt")
        if txt_file is not None:
            resume_text = txt_file.read().decode("utf-8", errors="ignore")
            st.text_area("Preview", resume_text, height=200, disabled=True)

    analyze_clicked = st.button("Analyze resume", type="primary", use_container_width=True)

with right:
    st.image(asset("analyze_icon.png"), width=55)
    st.markdown("#### What happens next")
    st.write(
        "Your text goes through the same cleanup as the training data — "
        "HTML stripped, punctuation removed, contractions expanded, and "
        "common stopwords filtered out — before the model estimates the "
        "closest-matching category."
    )

st.write("")

if analyze_clicked:
    if not resume_text or not resume_text.strip():
        st.markdown(
            '<div class="card card-gold">Please paste or upload some resume '
            "text first, so there's something to analyze.</div>",
            unsafe_allow_html=True,
        )
    else:
        cleaned = clean_text(resume_text)
        vector = pipeline["tfidf"].transform([cleaned])
        pred = pipeline["model"].predict(vector)[0]
        proba = pipeline["model"].predict_proba(vector)[0]
        predicted_label = pipeline["encoder"].inverse_transform([pred])[0]
        confidence = float(np.max(proba)) * 100

        res_col, img_col = st.columns([3, 1])
        with res_col:
            st.markdown(
                f"""
                <div class="result-box">
                    <div style="font-size:1rem; opacity:0.9;">Predicted category</div>
                    <h1>{predicted_label}</h1>
                    <div style="font-size:1rem; opacity:0.9;">Confidence: {confidence:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with img_col:
            st.image(asset("success_icon.png"), width=120)

        # Top-5 probability breakdown, multi-coloured
        st.write("")
        st.markdown("##### How the model weighed the top categories")
        proba_df = pd.DataFrame(
            {"category": pipeline["encoder"].classes_, "probability": proba}
        ).sort_values("probability", ascending=False).head(5)
        proba_chart = (
            alt.Chart(proba_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("probability:Q", axis=alt.Axis(format="%"), title="Probability"),
                y=alt.Y("category:N", sort="-x", title=None),
                color=alt.Color("category:N", legend=None, scale=alt.Scale(scheme="turbo")),
                tooltip=["category", alt.Tooltip("probability:Q", format=".1%")],
            )
            .properties(height=260)
        )
        st.altair_chart(proba_chart, use_container_width=True)

        st.caption(
            "This is a simple TF-IDF + Naive Bayes model — treat the result as a "
            "helpful first impression, not the final word, on any given resume."
        )

st.markdown(
    '<div class="footer-note">Built with Streamlit • TF-IDF • Multinomial Naive Bayes</div>',
    unsafe_allow_html=True,
)
