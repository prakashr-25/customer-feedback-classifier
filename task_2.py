import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

torch.manual_seed(42)

data = load_breast_cancer(as_frame=True)
df = data.frame
df.to_csv("data/breast_cancer.csv", index=False)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["target"]
)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_val = val_df.drop(columns=["target"])
y_val = val_df["target"]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

train_csv = pd.DataFrame(X_train)
train_csv["target"] = y_train.to_numpy()
train_csv.to_csv("data/cancer_train.csv", index=False)

val_csv = pd.DataFrame(X_val)
val_csv["target"] = y_val.to_numpy()
val_csv.to_csv("data/cancer_val.csv", index=False)

class CsvDataset(Dataset):

    def __init__(self, file):
        df = pd.read_csv(file)

        self.X = torch.tensor(
            df.drop(columns=["target"]).to_numpy(),
            dtype=torch.float32
        )

        self.y = torch.tensor(
            df["target"].to_numpy(),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

train_data = CsvDataset("data/cancer_train.csv")
val_data = CsvDataset("data/cancer_val.csv")

train_loader = DataLoader(
    train_data, batch_size=32, shuffle=True
)

val_loader = DataLoader(
    val_data, batch_size=32, shuffle=False
)

print("Training samples:", len(train_data))
print("Validation samples:", len(val_data))

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(30, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

model = MLP()

params = sum(p.numel() for p in model.parameters())
print("Params:", params)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_accuracy = 0
best_epoch = 0
patience = 8
counter = 0

train_losses = []
val_accuracies = []

for epoch in range(40):

    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        output = model(X_batch)
        loss = criterion(output, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    avg_loss = total_loss / len(train_data)
    train_losses.append(avg_loss)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:

            output = model(X_batch)
            prediction = (torch.sigmoid(output) >= 0.5).float()

            correct += (prediction == y_batch).sum().item()
            total += len(y_batch)

    accuracy = correct / total
    val_accuracies.append(accuracy)

    print(
        f"Epoch {epoch+1}: "
        f"Loss={avg_loss:.4f}, "
        f"Val Accuracy={accuracy:.4f}"
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch + 1
        counter = 0

        torch.save(model.state_dict(), "models/best_mlp.pt")

    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break

print("\nBest Val Accuracy:", round(best_accuracy, 4))
print("Best Epoch:", best_epoch)

loaded_model = MLP()
loaded_model.load_state_dict(
    torch.load("models/best_mlp.pt", weights_only=True)
)
loaded_model.eval()

correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in val_loader:

        output = loaded_model(X_batch)
        prediction = (torch.sigmoid(output) >= 0.5).float()

        correct += (prediction == y_batch).sum().item()
        total += len(y_batch)

reloaded_accuracy = correct / total

print("Reloaded Model Accuracy:", round(reloaded_accuracy, 4))
print("Identical:", reloaded_accuracy == best_accuracy)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_val)

lr_accuracy = accuracy_score(y_val, lr_pred)

print("Logistic Regression Accuracy:", round(lr_accuracy, 4))