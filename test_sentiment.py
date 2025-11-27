import joblib

# Load saved model + vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------------------
# TEST CASES (add as many as you want)
# ---------------------------


test_sentences = [

    # NEGATION → POSITIVE
    "the water supply is not dirty",
    "the drainage is not blocked anymore",
    "the roads are not bad today",
    "the metro is not delayed today",
    "the service is not terrible",

    # NEGATION → NEGATIVE
    "the park is not clean today",
    "the streetlights are not bright today",
    "the footpath is not safe",
    "the drainage system is not functioning well",
    "the bus service is not good today",
]






# ---------------------------
# BATCH PREDICTION
# ---------------------------
X = vectorizer.transform(test_sentences)
preds = model.predict(X)

print("\n------ SENTIMENT TEST RESULTS ------\n")
for text, label in zip(test_sentences, preds):
    print(f"{text}  -->  {label}")
