# app_simple.py
# Run: streamlit run app_simple.py
import re, html, warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---------- Normalization (must match training) ----------
EMOJI_MAP = {"🍔":" burger ", "🔥":" fire ", "🎬":" movie "}
SLANG = {
    "u":"you","r":"are","2day":"today","tmrw":"tomorrow","idk":"i dont know",
    "lol":"funny","smh":"disappointed","pm":"prime minister"
}
POLITICS_HINTS = {
    r"\bparl(iament)?\b": " parliament ",
    r"\belection(s)?\b": " election ",
    r"\bminister\b": " minister ",
    r"\bspeech\b": " speech ",
}

def normalize(s: str) -> str:
    s = html.unescape(str(s))
    s = s.lower()
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)  # speeech -> speech
    if SLANG:
        s = re.sub(r"\b(" + "|".join(map(re.escape, SLANG.keys())) + r")\b",
                   lambda m: SLANG[m.group(0)], s)
    s = "".join(EMOJI_MAP.get(ch, ch) for ch in s)    # map emojis to words
    s = re.sub(r"http\S+|www\.\S+", " ", s)           # drop urls
    s = re.sub(r"[^a-z0-9\s]", " ", s)                # keep alnum/space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def enrich_politics(s: str) -> str:
    for pat, rep in POLITICS_HINTS.items():
        s = re.sub(pat, rep, s)
    return s

def normalize_plus(s: str) -> str:
    return enrich_politics(normalize(s))

# ---------- UI ----------
st.set_page_config(page_title="Text Classifier", layout="centered")
st.title("🔮 Sentence Classifier")
st.caption("Type any sentence. The model predicts one of: **sports, politics, tech, food, entertainment**.")

# Paths
MODEL_PATH_DEFAULT = "text_cls_stacked.joblib"
csv_help = st.sidebar.expander("Optional: Train from CSV if model is missing")
with csv_help:
    st.write("If you don't have a saved model yet, point to your dataset (columns: `text`, `label`).")
    csv_path = st.text_input("CSV path (optional):", value="", placeholder="your_dataset.csv")
    train_btn = st.button("Train & Save Model (once)")

# ---------- Load or train model ----------
@st.cache_resource(show_spinner=False)
def load_model_or_train(path_model: str, csv_path: str | None):
    if os.path.exists(path_model):
        return joblib.load(path_model), None

    if not csv_path or not os.path.exists(csv_path):
        return None, "Model not found and no valid CSV provided."

    # Minimal training fallback (same pipeline as before)
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.naive_bayes import ComplementNB
    from sklearn.linear_model import SGDClassifier, LogisticRegression
    from sklearn.ensemble import StackingClassifier

    df = pd.read_csv(csv_path)[["text", "label"]].dropna().drop_duplicates().reset_index(drop=True)

    def fingerprint(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\d+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return " ".join([w for w in s.split() if len(w) > 3])

    df["group"] = df["text"].map(fingerprint)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(gss.split(df["text"], df["label"], groups=df["group"]))
    train_df = df.iloc[train_idx].copy()

    train_df["norm"] = train_df["text"].map(normalize_plus)

    word_tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=30000, min_df=2, lowercase=True)
    char_tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=30000, min_df=2, lowercase=True)
    features = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])

    svc_cal = CalibratedClassifierCV(LinearSVC(), method="isotonic", cv=3)
    nb      = ComplementNB()
    sgd     = SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-3, random_state=42)

    stack = StackingClassifier(
        estimators=[("svc", svc_cal), ("nb", nb), ("sgd", sgd)],
        final_estimator=LogisticRegression(max_iter=2000, random_state=42),
        stack_method="predict_proba",
        passthrough=False
    )
    pipe = Pipeline([("feat", features), ("clf", stack)])
    pipe.fit(train_df["norm"], train_df["label"])
    joblib.dump(pipe, path_model)
    return pipe, None

model = None
err = None
if train_btn:
    model, err = load_model_or_train(MODEL_PATH_DEFAULT, csv_path)

if model is None and err is None:
    # Try to load an existing model by default
    model, err = load_model_or_train(MODEL_PATH_DEFAULT, "")

if err:
    st.error(err)
    st.info("Upload or point to a CSV in the sidebar to train a model.")
elif model is None:
    st.stop()

# ---------- Inference box ----------
txt = st.text_area(
    "Enter a sentence:",
    value="PM gave big speeech 2day",
    help="Write anything: sports scores, political speech, smartphone updates, food reviews, movies…"
)

if st.button("Classify"):
    norm = normalize_plus(txt)
    pred = model.predict([norm])[0]

    # Show top-3 probabilities if available
    proba_md = ""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([norm])[0]
        classes = model.classes_
        top = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
        proba_md = "\n".join([f"- **{c}**: {p:.3f}" for c, p in top])

    st.success(f"**Prediction:** {pred}")
    if proba_md:
        st.markdown("**Top probabilities:**")
        st.markdown(proba_md)
