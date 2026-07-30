# Resume Category Classifier — Streamlit App

A colourful, humanized front-end for a TF-IDF + Multinomial Naive Bayes
resume classifier. No emojis are used anywhere — all icons are small custom
illustration PNGs bundled in `assets/`, so the app looks the same for every
user and needs no internet access to render them.

## Folder contents

```
resume_app/
├── app.py              # the Streamlit app
├── requirements.txt    # Python dependencies
├── make_icons.py       # (optional) regenerates the icon images
├── assets/             # banner + icon images used in the UI
└── Resume.csv          # <- add your dataset here (not included)
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Add your dataset

Place your `Resume.csv` file in the same folder as `app.py`. It should be
the same format used in your original notebook: a CSV where, after the
header row, the columns split out to include `resume_text` and `category`.

If you'd rather not keep the file locally, you can also upload it from the
app's sidebar at runtime — nothing needs to be hardcoded.

## 3. Run the app

```bash
streamlit run app.py
```

Streamlit will open the app in your browser (usually at
`http://localhost:8501`).

## What it does

1. Loads and cleans your resume dataset (HTML stripped, punctuation
   removed, contractions expanded, stopwords dropped).
2. Vectorizes the cleaned text with TF-IDF (top 3000 features).
3. Trains a Multinomial Naive Bayes classifier and reports test accuracy.
4. Lets you paste in a new resume (or upload a `.txt` file) and shows:
   - the predicted category, in a colourful result card
   - the model's confidence
   - a chart of the top-5 most likely categories

## Notes on deployment

- Training happens once per session and is cached (`st.cache_resource`), so
  re-running predictions is instant after the first load.
- To deploy on **Streamlit Community Cloud**: push this folder (including
  `Resume.csv` if your dataset license allows it, or wire up the uploader)
  to a GitHub repo, then point Streamlit Cloud at `app.py`.
- The GloVe embedding step from the original notebook was left out of the
  deployed app on purpose — downloading a ~128MB embedding file on every
  app boot is slow and unnecessary for the TF-IDF + Naive Bayes pipeline
  that's actually used for the final prediction. It's easy to add back in
  as an alternate vectorization option if you'd like.
