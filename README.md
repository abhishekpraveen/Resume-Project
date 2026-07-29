# ResumeIQ — NLP Resume Intelligence Studio

A Streamlit app built from your original NLP notebook (text cleaning →
tokenization → stemming/lemmatization → POS tagging → TF-IDF → Naive Bayes
job-role classifier). It adds file upload, a richer training set across
6 roles, and visual analytics — with icon/image-based UI instead of emojis.

## What it does

1. Upload a resume as `.txt`, `.docx`, or `.pdf`.
2. Watch it move through the same cleaning pipeline as your notebook
   (HTML strip → punctuation removal → contraction fixing → lowercasing →
   number removal → stopword filtering → tokenization → stemming →
   lemmatization → POS tagging), with a plain-language explanation at each step.
3. See a keyword cloud, top keywords, and a POS-tag distribution chart.
4. Get a predicted job role from a TF-IDF + Multinomial Naive Bayes model,
   trained on an expanded sample dataset covering Data Scientist, Java
   Developer, Web Developer, DevOps Engineer, Business Analyst, and Machine
   Learning Engineer — with a confidence chart across all roles.

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads a few small NLTK data packages (punkt, stopwords,
wordnet, POS tagger) automatically and caches them.

## Deploy on Streamlit Community Cloud (free)

1. Push `app.py`, `requirements.txt`, and this `README.md` to a public
   (or private) GitHub repository.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click **New app**, pick the repo/branch, and set the main file to `app.py`.
4. Click **Deploy**. First boot takes ~1–2 minutes while dependencies and
   NLTK data install.

## Deploy elsewhere (Render, Railway, a VPS, etc.)

Any host that can run a long-lived Python process works:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Improving the classifier

The demo trains on a small hand-written dataset (`TRAINING_DATA` dict near
the top of `app.py`) so it works out of the box with no external files.
For real use, replace it with a proper labeled dataset of resumes: swap the
`TRAINING_DATA` dict for a CSV load (`pandas.read_csv`) of `(text, role)`
pairs, feed it through `run_pipeline()` for consistent cleaning, then refit
the vectorizer and model the same way `train_model()` does now.

## File overview

- `app.py` — the full Streamlit application (pipeline, model, UI).
- `requirements.txt` — pinned minimum versions for deployment.
- `README.md` — this file.
