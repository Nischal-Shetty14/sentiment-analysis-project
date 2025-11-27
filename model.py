# model.py - FIXED VERSION
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re

def expand_contractions(text):
    mapping = {
        "isn't": "is not", "wasn't": "was not", "aren't": "are not", 
        "don't": "do not", "didn't": "did not", "can't": "can not", 
        "won't": "will not", "haven't": "have not", "hasn't": "has not", 
        "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not"
    }
    pattern = re.compile("|".join(re.escape(k) for k in mapping.keys()))
    return pattern.sub(lambda m: mapping[m.group(0)], text)

def clean_text(text):
    """
    Minimal cleaning - DON'T remove stopwords for sentiment!
    Stopwords like 'not', 'is', 'are' are critical for sentiment.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = expand_contractions(text)
    # Remove special chars but keep spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Clean extra whitespace
    text = " ".join(text.split())
    return text

# Load and prepare data
df = pd.read_csv("dataset.csv")

# Clean the text
X = df["text"].apply(clean_text)
y = df["label"]

print(f"Dataset size: {len(df)}")
print(f"Label distribution:\n{y.value_counts()}\n")

# TF-IDF with improved parameters
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),      # unigrams and bigrams
    min_df=1,                # ignore very rare terms
    max_df=0.95,             # ignore very common terms     # limit vocabulary size
    strip_accents='unicode',
    lowercase=True
)

X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}\n")

# Train model
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    C=1.0,                   # regularization strength
    solver='lbfgs',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

# Detailed evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model and vectorizer saved!")