"""
ResumeIQ — An NLP-powered Resume Intelligence Studio
------------------------------------------------------
Built from an original NLP pipeline (cleaning, tokenization, stemming,
lemmatization, POS tagging, TF-IDF) and extended into a full, deployable
Streamlit application with a job-role classifier and visual analytics.

Run locally:
    streamlit run app.py

Deploy:
    Push this folder to a GitHub repo and deploy on Streamlit Community
    Cloud (https://streamlit.io/cloud) pointing at app.py, or run it on
    any server with `streamlit run app.py --server.port $PORT`.
"""

import io
import re
import string
import random
from collections import Counter

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")  # headless backend, required for server deployment
import matplotlib.pyplot as plt

import nltk
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    import contractions
    HAS_CONTRACTIONS = True
except ImportError:
    HAS_CONTRACTIONS = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False


# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="ResumeIQ | NLP Resume Studio",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# ICONS — inline SVG (no emojis, no external network calls)
# =============================================================
ICONS = {
    "document": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>""",
    "brain": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44
        2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>
        <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58
        2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/></svg>""",
    "chart": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/>
        <line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>""",
    "target": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>
        <circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>""",
    "cloud": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h.79a4.5 4.5 0 1 1 0 9Z"/></svg>""",
    "briefcase": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2"/>
        <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>""",
    "upload": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>""",
    "sparkle": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 4.5L5.5 9.4l4.6 1.9L12 15.8l1.9-4.5
        4.6-1.9-4.6-1.9L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>""",
    "layers": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/>
        <polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>""",
    "tag": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41 13.42 20.6a2 2 0 0 1-2.83 0L2 12.01V2h10.01l8.58 8.58a2 2 0 0 1 0 2.83Z"/>
        <line x1="7" y1="7" x2="7.01" y2="7"/></svg>""",
}


def icon_html(name, size=22, color="currentColor"):
    svg = ICONS.get(name, "")
    svg = svg.replace('stroke="currentColor"', f'stroke="{color}"')
    return f'<span style="display:inline-flex;width:{size}px;height:{size}px;vertical-align:middle;">{svg}</span>'


def section_header(icon, title, subtitle=""):
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-icon">{icon_html(icon, 26, "#ffffff")}</div>
            <div>
                <div class="section-title">{title}</div>
                <div class="section-sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================
# STYLING
# =============================================================
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

        html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

        .stApp {
            background: radial-gradient(circle at 10% 0%, #1b1035 0%, #0f0a24 35%, #0a0718 100%);
            color: #eae6ff;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1140 0%, #120c2b 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .hero {
            background: linear-gradient(120deg, #7b2ff7 0%, #f107a3 50%, #ff8a3d 100%);
            border-radius: 24px;
            padding: 38px 42px;
            margin-bottom: 28px;
            box-shadow: 0 20px 50px rgba(123,47,247,0.35);
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content: "";
            position: absolute; top: -60px; right: -60px;
            width: 220px; height: 220px; border-radius: 50%;
            background: rgba(255,255,255,0.10);
        }
        .hero h1 {
            font-size: 2.5rem; font-weight: 800; color: white; margin-bottom: 6px;
            letter-spacing: -0.5px;
        }
        .hero p { color: rgba(255,255,255,0.92); font-size: 1.05rem; max-width: 640px; }
        .hero-badge {
            display: inline-block; background: rgba(255,255,255,0.18);
            color: white; padding: 5px 14px; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600; letter-spacing: 0.4px;
            margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.3);
        }

        .glass-card {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 18px;
            padding: 22px 24px;
            margin-bottom: 18px;
            backdrop-filter: blur(6px);
        }

        .section-head { display:flex; align-items:center; gap:14px; margin: 6px 0 16px 0; }
        .section-icon {
            background: linear-gradient(135deg, #7b2ff7, #f107a3);
            width: 44px; height: 44px; border-radius: 12px;
            display:flex; align-items:center; justify-content:center;
            box-shadow: 0 6px 16px rgba(123,47,247,0.4);
        }
        .section-title { font-size: 1.25rem; font-weight: 700; color: #fff; }
        .section-sub { font-size: 0.85rem; color: #b9b3d9; }

        .pill {
            display:inline-block; padding: 6px 14px; margin: 4px 6px 4px 0;
            border-radius: 999px; font-size: 0.82rem; font-weight: 600;
            background: linear-gradient(135deg, rgba(123,47,247,0.25), rgba(241,7,163,0.25));
            border: 1px solid rgba(255,255,255,0.15); color: #f0eaff;
        }

        .result-card {
            background: linear-gradient(135deg, #10c29b 0%, #0aa1c9 100%);
            border-radius: 20px; padding: 28px 30px; text-align:center;
            box-shadow: 0 15px 40px rgba(16,194,155,0.30);
        }
        .result-role { font-size: 2.1rem; font-weight: 800; color: white; margin: 6px 0; }
        .result-label { color: rgba(255,255,255,0.85); font-size: 0.9rem; letter-spacing: 1px; text-transform: uppercase;}
        .result-conf { color: rgba(255,255,255,0.95); font-size: 1rem; margin-top: 8px; }

        .step-card {
            border-left: 3px solid #7b2ff7;
            background: rgba(255,255,255,0.03);
            padding: 14px 18px; border-radius: 0 14px 14px 0; margin-bottom: 10px;
        }
        .step-title { font-weight: 700; color: #d9b8ff; font-size: 0.95rem; margin-bottom: 4px;}
        .step-text { color: #cfc9ec; font-size: 0.87rem; }

        .footer-note {
            text-align:center; color: #7a749e; font-size: 0.8rem; margin-top: 40px;
            padding-top: 18px; border-top: 1px solid rgba(255,255,255,0.07);
        }

        .stButton>button {
            background: linear-gradient(135deg, #7b2ff7, #f107a3);
            color: white; border: none; border-radius: 12px; padding: 10px 22px;
            font-weight: 600; box-shadow: 0 8px 20px rgba(123,47,247,0.35);
        }
        .stButton>button:hover { opacity: 0.92; }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.04); border-radius: 16px;
            border: 1.5px dashed rgba(255,255,255,0.25);
        }

        code, .stCodeBlock { font-family: 'Fira Code', monospace !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================
# NLTK SETUP
# =============================================================
@st.cache_resource(show_spinner=False)
def ensure_nltk_data():
    packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, pkg in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    return True


# =============================================================
# TEXT EXTRACTION
# =============================================================
def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if name.endswith(".txt"):
        return raw_bytes.decode("utf-8", errors="ignore")

    if name.endswith(".docx"):
        if not HAS_DOCX:
            st.error("python-docx is not installed. Add it to requirements.txt.")
            return ""
        f = io.BytesIO(raw_bytes)
        d = docx.Document(f)
        return "\n".join(p.text for p in d.paragraphs)

    if name.endswith(".pdf"):
        if not HAS_PDF:
            st.error("pdfplumber is not installed. Add it to requirements.txt.")
            return ""
        f = io.BytesIO(raw_bytes)
        text_chunks = []
        with pdfplumber.open(f) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_chunks.append(t)
        return "\n".join(text_chunks)

    st.error("Unsupported file type. Please upload a .txt, .docx, or .pdf file.")
    return ""


# =============================================================
# NLP PIPELINE  (mirrors the original notebook, cleaned up)
# =============================================================
def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def fix_contractions(text):
    if HAS_CONTRACTIONS:
        try:
            return contractions.fix(text)
        except Exception:
            return text
    return text


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def remove_stopwords(text, stop_words):
    words = text.split()
    return " ".join(w for w in words if w not in stop_words)


def run_pipeline(raw_text):
    """Returns a dict capturing every intermediate stage for transparency."""
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag

    stages = {}
    stages["raw"] = raw_text

    text = clean_html(raw_text)
    stages["html_cleaned"] = text

    text = remove_punctuation(text)
    stages["no_punctuation"] = text

    text = fix_contractions(text)
    stages["contractions_fixed"] = text

    text = text.lower()
    stages["lowercased"] = text

    text = remove_numbers(text)
    stages["no_numbers"] = text

    stop_words = set(stopwords.words("english"))
    text_no_stop = remove_stopwords(text, stop_words)
    stages["no_stopwords"] = text_no_stop

    tokens = word_tokenize(text_no_stop)
    stages["tokens"] = tokens

    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(t) for t in tokens]
    stages["stemmed"] = stemmed

    pos = pos_tag(tokens)
    stages["pos_tags"] = pos

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    stages["lemmatized"] = lemmatized

    stages["final_text"] = " ".join(lemmatized)
    return stages


# =============================================================
# TRAINING DATA  (expanded well beyond the original 6-row toy set)
# =============================================================
TRAINING_DATA = {
    "Data Scientist": [
        "python machine learning data science sql pandas numpy scikit learn statistics",
        "deep learning neural network tensorflow keras pytorch data analysis visualization",
        "regression classification clustering feature engineering model evaluation cross validation",
        "data scientist predictive modeling exploratory data analysis jupyter notebook statistics",
        "nlp natural language processing text mining sentiment analysis python pandas",
        "big data spark hadoop data pipeline etl python sql analytics dashboard",
    ],
    "Java Developer": [
        "java spring boot hibernate microservices backend rest api mysql",
        "core java multithreading collections framework jdbc servlet jsp",
        "spring mvc dependency injection maven gradle unit testing junit",
        "java developer object oriented programming design patterns enterprise application",
        "kafka rabbitmq spring cloud microservices distributed systems java backend",
        "java se java ee application server tomcat weblogic soap rest",
    ],
    "Web Developer": [
        "html css javascript react frontend responsive design bootstrap",
        "web developer angular typescript node express mongodb full stack",
        "vue javascript webpack sass ui ux responsive web design",
        "react redux hooks component state management frontend javascript",
        "php laravel mysql html css javascript web application development",
        "next js react server side rendering tailwind css frontend engineer",
    ],
    "DevOps Engineer": [
        "devops docker kubernetes ci cd jenkins pipeline automation terraform",
        "aws azure cloud infrastructure ansible configuration management deployment",
        "linux shell scripting monitoring prometheus grafana devops engineer",
        "kubernetes container orchestration helm gitops infrastructure as code",
        "jenkins gitlab ci pipeline automation docker compose deployment aws",
        "site reliability engineering cloud infrastructure automation monitoring logging",
    ],
    "Business Analyst": [
        "business analyst requirements gathering stakeholder management documentation",
        "sql excel data analysis reporting dashboards power bi tableau",
        "business process improvement gap analysis use cases user stories",
        "agile scrum product backlog stakeholder communication business requirements",
        "financial analysis forecasting budgeting excel powerpoint reporting analyst",
        "market research competitive analysis kpi reporting business intelligence",
    ],
    "Machine Learning Engineer": [
        "machine learning engineer model deployment mlops docker kubernetes flask api",
        "tensorflow pytorch model training deployment production ml pipeline",
        "computer vision opencv deep learning cnn image classification pytorch",
        "ml engineer feature store model monitoring drift detection production",
        "recommendation system collaborative filtering machine learning production api",
        "mlops ci cd model versioning experiment tracking mlflow kubeflow",
    ],
}


@st.cache_resource(show_spinner=False)
def train_model():
    documents, labels = [], []
    for role, samples in TRAINING_DATA.items():
        for s in samples:
            documents.append(s)
            labels.append(role)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Refit on full data for the final deployed model (small dataset -> use it all)
    model_full = MultinomialNB()
    model_full.fit(X, y)

    return vectorizer, model_full, sorted(set(labels)), acc


# =============================================================
# VISUAL HELPERS
# =============================================================
def make_wordcloud_image(text):
    if not text.strip():
        return None
    wc = WordCloud(
        width=1000,
        height=500,
        background_color=None,
        mode="RGBA",
        colormap="cool",
        prefer_horizontal=0.9,
        max_words=80,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def probability_chart(labels, probs):
    order = np.argsort(probs)[::-1]
    labels = [labels[i] for i in order]
    probs = [probs[i] for i in order]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=labels,
            orientation="h",
            marker=dict(
                color=probs,
                colorscale=[[0, "#7b2ff7"], [1, "#f107a3"]],
                line=dict(width=0),
            ),
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=30, t=10, b=10),
        height=320,
        xaxis=dict(title="Confidence", range=[0, max(probs) * 1.25], showgrid=False),
        yaxis=dict(autorange="reversed"),
        font=dict(family="Poppins", color="#eae6ff"),
    )
    return fig


def pos_distribution_chart(pos_tags):
    tag_groups = {
        "Noun": {"NN", "NNS", "NNP", "NNPS"},
        "Verb": {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"},
        "Adjective": {"JJ", "JJR", "JJS"},
        "Adverb": {"RB", "RBR", "RBS"},
        "Other": set(),
    }

    def group_of(tag):
        for g, tags in tag_groups.items():
            if tag in tags:
                return g
        return "Other"

    counts = Counter(group_of(tag) for _, tag in pos_tags)
    df = pd.DataFrame({"Category": list(counts.keys()), "Count": list(counts.values())})
    df = df.sort_values("Count", ascending=False)

    fig = px.bar(
        df,
        x="Category",
        y="Count",
        color="Category",
        color_discrete_sequence=["#7b2ff7", "#f107a3", "#ff8a3d", "#10c29b", "#0aa1c9"],
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        font=dict(family="Poppins", color="#eae6ff"),
    )
    return fig


def top_keywords(lemmatized_tokens, n=15):
    counts = Counter(w for w in lemmatized_tokens if len(w) > 2)
    return counts.most_common(n)


# =============================================================
# APP LAYOUT
# =============================================================
def main():
    inject_css()
    ensure_nltk_data()
    vectorizer, model, class_labels, val_accuracy = train_model()

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                {icon_html('sparkle', 28, '#f0a6ff')}
                <span style="font-size:1.3rem;font-weight:800;color:white;">ResumeIQ</span>
            </div>
            <p style="color:#b9b3d9;font-size:0.85rem;margin-top:-4px;">
                An NLP studio that reads a resume the way a recruiter would —
                cleans it, understands it, and tells you which role it fits best.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            f"<div style='display:flex;gap:8px;align-items:center;color:#d9b8ff;font-weight:600;'>"
            f"{icon_html('layers', 18)} Pipeline stages</div>",
            unsafe_allow_html=True,
        )
        for step in [
            "HTML & noise cleanup",
            "Punctuation & digit removal",
            "Contraction expansion",
            "Stopword filtering",
            "Tokenization",
            "Stemming & Lemmatization",
            "POS tagging",
            "TF-IDF vectorization",
            "Naive Bayes classification",
        ]:
            st.markdown(f"<div style='color:#cfc9ec;font-size:0.83rem;padding:3px 0;'>› {step}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            f"<div style='color:#b9b3d9;font-size:0.8rem;'>Classifier validation accuracy on held-out samples: "
            f"<b style='color:#10c29b;'>{val_accuracy*100:.0f}%</b> "
            f"(small demo dataset — accuracy improves with more training resumes).</div>",
            unsafe_allow_html=True,
        )

    # ---------- HERO ----------
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-badge">NLP · TF-IDF · Naive Bayes</div>
            <h1>Upload a resume. Watch the language model work.</h1>
            <p>ResumeIQ takes a raw resume, walks it through a full text-preprocessing
            pipeline, then predicts the job role it best fits — with visual, human-readable
            explanations at every step.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- UPLOAD ----------
    section_header("upload", "Bring in a resume", "Supports .txt, .docx and .pdf")
    uploaded = st.file_uploader(
        "Drop a resume file here",
        type=["txt", "docx", "pdf"],
        label_visibility="collapsed",
    )

    demo_mode = False
    if uploaded is None:
        demo_mode = st.checkbox("No file handy — try it with a sample resume snippet")

    raw_text = ""
    if uploaded is not None:
        raw_text = extract_text(uploaded)
    elif demo_mode:
        raw_text = (
            "Abhishek is a data enthusiast with hands-on experience in Python, "
            "Machine Learning, and SQL. Skilled in building predictive models using "
            "scikit-learn, pandas and numpy, with a strong foundation in statistics "
            "and data visualization. Completed multiple projects in Natural Language "
            "Processing and enjoys turning messy data into clear insights."
        )

    if not raw_text.strip():
        st.info("Upload a resume, or try the sample snippet above, to see the full analysis.")
        st.markdown('<div class="footer-note">Built with Streamlit · scikit-learn · NLTK</div>', unsafe_allow_html=True)
        return

    # ---------- RUN PIPELINE ----------
    with st.spinner("Reading and understanding the resume..."):
        stages = run_pipeline(raw_text)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("document", "Original text", "Exactly what was extracted from the file")
    with st.expander("Show raw extracted text", expanded=False):
        st.text_area("Raw text", stages["raw"], height=180, label_visibility="collapsed")

    # ---------- PIPELINE WALKTHROUGH ----------
    section_header("layers", "How the text was cleaned", "Step-by-step, in plain language")

    step_explanations = [
        ("Stripping hidden markup", "Removed any leftover HTML tags so only readable text remains.", stages["html_cleaned"]),
        ("Removing punctuation", "Commas, periods and symbols are dropped — they add noise, not meaning, for the model.", stages["no_punctuation"]),
        ("Expanding contractions", "Words like \"don't\" become \"do not\" so the model reads consistent, full forms.", stages["contractions_fixed"]),
        ("Lowercasing & removing numbers", "Everything is normalized to lowercase and digits are stripped out.", stages["no_numbers"]),
        ("Filtering stopwords", "Common filler words (\"the\", \"and\", \"is\"...) are removed to keep only meaningful terms.", stages["no_stopwords"]),
    ]
    for title, desc, _ in step_explanations:
        st.markdown(
            f"""<div class="step-card"><div class="step-title">{title}</div>
            <div class="step-text">{desc}</div></div>""",
            unsafe_allow_html=True,
        )

    with st.expander("See the cleaned text after every stage"):
        for title, _, content in step_explanations:
            st.markdown(f"**{title}**")
            st.code(content[:1500] if isinstance(content, str) else str(content)[:1500], language="text")

    tokens = stages["tokens"]
    stemmed = stages["stemmed"]
    lemmatized = stages["lemmatized"]
    pos_tags = stages["pos_tags"]
    final_text = stages["final_text"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_header("brain", "Tokens & Stems", f"{len(tokens)} tokens extracted")
        st.write(", ".join(tokens[:60]) + (" ..." if len(tokens) > 60 else ""))
        st.caption("Sample of stemmed forms")
        st.code(" ".join(stemmed[:40]), language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_header("chart", "Part-of-Speech mix", "Grammatical role of each word")
        st.plotly_chart(pos_distribution_chart(pos_tags), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- KEYWORDS + WORDCLOUD ----------
    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns([1, 1.3])
    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_header("tag", "Top keywords", "Most frequent meaningful terms")
        kws = top_keywords(lemmatized)
        pills = "".join(f'<span class="pill">{w} · {c}</span>' for w, c in kws)
        st.markdown(pills or "<i>No significant keywords found.</i>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_header("cloud", "Keyword cloud", "A visual fingerprint of the resume")
        img = make_wordcloud_image(final_text)
        if img:
            st.image(img, use_container_width=True)
        else:
            st.write("Not enough text to build a keyword cloud.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- CLASSIFICATION ----------
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("target", "Predicted job role", "Based on a TF-IDF + Naive Bayes classifier")

    vec = vectorizer.transform([final_text])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    colA, colB = st.columns([1, 1.4])
    with colA:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Best-fit role</div>
                <div class="result-role">{prediction}</div>
                <div class="result-conf">Confidence: {max(proba)*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with colB:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_header("briefcase", "Role probabilities", "How confident the model is across every role")
        st.plotly_chart(probability_chart(class_labels, list(proba)), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        "Download cleaned & lemmatized text",
        data=final_text,
        file_name="cleaned_resume.txt",
        mime="text/plain",
    )

    st.markdown(
        '<div class="footer-note">ResumeIQ · An NLP pipeline turned into a product · '
        'Built with Streamlit, scikit-learn, NLTK and a little bit of design care.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
