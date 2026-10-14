import pandas as pd
import re
from sklearn.model_selection import train_test_split

subjects = [
    "the product", "the delivery", "the website",
    "the mobile app", "the support team", "the checkout",
    "the packaging", "the order", "the subscription",
    "the payment", "the service", "the refund process",
    "the booking", "the account setup", "the shopping experience"
]

positive = [
    "great", "excellent", "amazing", "wonderful", "fantastic",
    "good", "pleasant", "smooth", "reliable", "impressive",
    "satisfying", "delightful", "outstanding", "superb", "enjoyable"
]

negative = [
    "bad", "terrible", "awful", "horrible", "poor",
    "unpleasant", "frustrating", "unreliable", "disappointing",
    "unsatisfying", "annoying", "dreadful", "negative",
    "mediocre", "painful"
]

texts = []
labels = []

for subject in subjects:

    for word in positive:
        texts.append(f"My experience with {subject} was {word}!")
        labels.append("positive")

    for word in negative:
        texts.append(f"My experience with {subject} was {word}.")
        labels.append("negative")

df = pd.DataFrame({"text": texts, "label": labels})

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

df = df.drop_duplicates(subset=["clean_text"])

train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["label"]
)

df.to_csv("data/feedback.csv", index=False)
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Total records:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

print("\nTrain:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

print("\nExample tokens:", train.iloc[0]["clean_text"].split())