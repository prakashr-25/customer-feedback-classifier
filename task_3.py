import time
import numpy as np
import pandas as pd
import gensim.downloader as api

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_train = train["clean_text"]
y_train = train["label"]

X_test = test["clean_text"]
y_test = test["label"]

baseline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression(max_iter=1000))
])

start = time.perf_counter()
baseline.fit(X_train, y_train)
train_time = time.perf_counter() - start

start = time.perf_counter()
predictions = baseline.predict(X_test)
inference_time = (time.perf_counter() - start) * 1000 / len(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")

print("TF-IDF Baseline--")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test, predictions,
    labels=["negative", "positive"]
))

features = baseline.named_steps["tfidf"].get_feature_names_out()
weights = baseline.named_steps["model"].coef_[0]

negative_words = np.argsort(weights)[:10]
positive_words = np.argsort(weights)[-10:][::-1]

print("\nTop 10 Negative Words:")
print(features[negative_words])

print("\nTop 10 Positive Words:")
print(features[positive_words])

results = pd.DataFrame([{
    "Model": "TF-IDF + Logistic Regression",
    "Accuracy": accuracy,
    "Macro F1": macro_f1,
    "Training time (s)": train_time,
    "Inference time per ticket (ms)": inference_time
}])

results.to_csv("outputs/tfidf_results.csv", index=False)

print("\nLoading GloVe embeddings...")
glove = api.load("glove-wiki-gigaword-50")

for word in ["king", "computer"]:
    print("\nNearest words to", word)

    for similar, score in glove.most_similar(word, topn=5):
        print(similar, round(score, 4))

analogy = glove.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=5
)

print("\nking - man + woman:")
print(analogy)

def sentence_vector(sentence):
    words = sentence.split()
    vectors = []

    for word in words:
        if word in glove:
            vectors.append(glove[word])

    if len(vectors) == 0:
        return np.zeros(50)

    return np.mean(vectors, axis=0)

X_train_vectors = np.array([
    sentence_vector(text) for text in X_train
])

X_test_vectors = np.array([
    sentence_vector(text) for text in X_test
])

embedding_model = LogisticRegression(max_iter=1000)

embedding_model.fit(X_train_vectors, y_train)
embedding_pred = embedding_model.predict(X_test_vectors)

embedding_accuracy = accuracy_score(y_test, embedding_pred)
embedding_f1 = f1_score(y_test, embedding_pred, average="macro")

print("\nGloVe Sentence Embeddings--")
print("Accuracy:", round(embedding_accuracy, 4))
print("Macro F1:", round(embedding_f1, 4))

print("\nComparison--")
print("TF-IDF Accuracy:", round(accuracy, 4))
print("GloVe Accuracy:", round(embedding_accuracy, 4))