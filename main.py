from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Customer Feedback Classifier")

model_path = "models/distilbert_sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

class Feedback(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("Text cannot be empty")
        return value

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(feedback: Feedback):

    inputs = tokenizer(
        feedback.text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    predicted_id = torch.argmax(probabilities).item()

    label = model.config.id2label[predicted_id]
    confidence = probabilities[predicted_id].item()

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }