# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib, os, re
import pandas as pd
import nltk
from nltk.corpus import stopwords



ROOT = os.path.dirname(__file__) or "."
MODEL_PATH = os.path.join(ROOT, "model.pkl")
VECT_PATH = os.path.join(ROOT, "vectorizer.pkl")
DATA_PATH = os.path.join(ROOT, "dataset.csv")

app = Flask(__name__)

# load model & vect
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# prep stopwords - keep not/no
stop = set(stopwords.words("english"))
stop.discard("not"); stop.discard("no")

def expand_contractions(text):
    mapping = {
        "isn't": "is not", "wasn't": "was not", "aren't": "are not", "don't": "do not",
        "didn't": "did not", "can't": "can not", "won't": "will not", "haven't": "have not",
        "hasn't": "has not", "couldn't": "could not", "wouldn't": "would not"
    }
    pattern = re.compile("|".join(re.escape(k) for k in mapping.keys()))
    return pattern.sub(lambda m: mapping[m.group(0)], text)


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text
def apply_but_rule(text):
    """
    Handles contrastive sentences of the form:
    X but Y  →  return Y (dominant sentiment)
    """
    parts = text.split(" but ")
    if len(parts) > 1:
        return parts[-1].strip()  # take clause after last 'but'
    return text


# --- Category Detection for Dataset Charts ---
df_full = pd.read_csv(DATA_PATH)

def detect_category(text):
    t = text.lower()
    if "road" in t: return "Roads"
    if "streetlight" in t or "street light" in t: return "Streetlights"
    if "water" in t: return "Water Supply"
    if "garbage" in t: return "Garbage Collection"
    if "park" in t: return "Parks"
    if "bus" in t: return "Bus Service"
    if "metro" in t: return "Metro"
    if "drainage" in t: return "Drainage"
    if "footpath" in t: return "Footpaths"
    if "traffic" in t: return "Traffic Signals"
    return "Other"

df_full["category"] = df_full["text"].apply(detect_category)
CATEGORY_ORDER = [
    "Roads", "Streetlights", "Water Supply", "Garbage Collection",
    "Parks", "Bus Service", "Metro", "Drainage", "Footpaths",
    "Traffic Signals", "Other"
]

raw_counts = df_full["category"].value_counts().to_dict()

CATEGORY_COUNTS = {cat: raw_counts.get(cat, 0) for cat in CATEGORY_ORDER}




@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probs = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("user_text", "")
        cleaned = clean_text(input_text)
        cleaned = apply_but_rule(cleaned)
        vec = vectorizer.transform([cleaned])
        proba = model.predict_proba(vec)[0]  # order corresponds to model.classes_
        classes = model.classes_
        probs = list(zip(classes, proba.tolist()))
        # sort by prob desc
        probs = sorted(probs, key=lambda x: x[1], reverse=True)
        prediction = probs[0][0]
        confidence = probs[0][1]
        # If confidence low, we can show 'Uncertain' (threshold 0.5)
        if confidence < 0.5:
            result = f"Uncertain — best guess: {prediction} ({confidence:.2f})"
        else:
            result = f"{prediction} ({confidence:.2f})"
    return render_template(
        "index.html",
        result=result,
        probs=probs,
        text=input_text,
        category_counts=CATEGORY_COUNTS
    )

@app.route("/dashboard")
def dashboard():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        counts = df['label'].astype(str).str.lower().value_counts().to_dict()
    else:
        counts = {}
    return render_template("dashboard.html", counts=counts)

# simple JSON endpoint for AJAX predictions
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    text = data.get("text","")
    cleaned = clean_text(text)
    cleaned = apply_but_rule(cleaned)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    classes = model.classes_.tolist()
    # prepare sorted probs
    probs = sorted(list(zip(classes, proba.tolist())), key=lambda x: x[1], reverse=True)
    return jsonify({"probs": probs})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

