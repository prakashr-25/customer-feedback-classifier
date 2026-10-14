import time
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

sentences = [
    "The customer support was excellent",
    "The delivery was terrible"
]

print("Tokenizer Output--")

for sentence in sentences:
    encoded = tokenizer(sentence)

    print("\nSentence:", sentence)
    print("Tokens:", tokenizer.convert_ids_to_tokens(
        encoded["input_ids"]
    ))
    print("Input IDs:", encoded["input_ids"])
    print("Attention Mask:", encoded["attention_mask"])


sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

sentiment_sentences = [
    "This product is amazing",
    "The service is very bad",
    "I really enjoyed the delivery experience"
]

print("\nSentiment Pipeline--")

for sentence in sentiment_sentences:
    result = sentiment_pipeline(sentence)[0]
    print(sentence, "->", result)

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

ner_sentences = [
    "Ravi works at Google in Chennai",
    "Priya visited Microsoft in Bangalore",
    "Arun works for Amazon in India"
]

print("\nNer Pipeline--")

for sentence in ner_sentences:
    print("\nSentence:", sentence)

    for entity in ner_pipeline(sentence):
        print(
            entity["word"],
            entity["entity_group"],
            round(float(entity["score"]), 4)
        )


train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

label_map = {
    "negative": 0,
    "positive": 1
}

class FeedbackDataset(Dataset):

    def __init__(self, df):
        self.encodings = tokenizer(
            df["clean_text"].tolist(),
            truncation=True,
            padding=True,
            max_length=64
        )

        self.labels = [
            label_map[label] for label in df["label"]
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {}

        for key in self.encodings:
            item[key] = torch.tensor(
                self.encodings[key][index]
            )

        item["labels"] = torch.tensor(self.labels[index])
        return item

train_data = FeedbackDataset(train)
val_data = FeedbackDataset(val)
test_data = FeedbackDataset(test)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1}
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, average="macro"
        )
    }

training_args = TrainingArguments(
    output_dir="models/checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

print("\nFine-Tuning--")

start = time.perf_counter()
trainer.train()
train_time = time.perf_counter() - start

val_results = trainer.evaluate()
print("\nValidation:", val_results)

test_results = trainer.predict(test_data)

accuracy = test_results.metrics["test_accuracy"]
macro_f1 = test_results.metrics["test_macro_f1"]

print("\nDistilBERT Results--")
print("Test Accuracy:", round(accuracy, 4))
print("Test Macro F1:", round(macro_f1, 4))


model_path = "models/distilbert_sentiment"

trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

print("\nModel saved to:", model_path)


trained_model = trainer.model
trained_model.eval()

device = trained_model.device
texts = test["clean_text"].tolist()

if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

with torch.no_grad():

    for i in range(0, len(texts), 8):

        batch = texts[i:i+8]

        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        trained_model(**inputs)

if device.type == "cuda":
    torch.cuda.synchronize()

inference_time = (
    time.perf_counter() - start
) * 1000 / len(texts)


tfidf_results = pd.read_csv("outputs/tfidf_results.csv")

bert_results = pd.DataFrame([{
    "Model": "Fine-tuned DistilBERT",
    "Accuracy": accuracy,
    "Macro F1": macro_f1,
    "Training time (s)": train_time,
    "Inference time per ticket (ms)": inference_time
}])

comparison = pd.concat(
    [tfidf_results, bert_results],
    ignore_index=True
)

comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nFinal model comparison--")
print(comparison.round(4).to_string(index=False))