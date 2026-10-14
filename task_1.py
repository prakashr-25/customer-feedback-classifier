import numpy as np
import matplotlib.pyplot as plt
import torch

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)

def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)

def cross_entropy(actual, predicted):
    predicted = np.clip(predicted, 1e-8, 1 - 1e-8)
    return -np.mean(
        actual * np.log(predicted)
        + (1 - actual) * np.log(1 - predicted)
    )

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

print("Softmax([2.0, 1.0, 0.1]):", np.round(softmax(np.array([2.0, 1.0, 0.1])), 3))
print("MSE example:", mse(np.array([1, 0]), np.array([0.8, 0.2])))


w, x, b, t = 1, 2, 0, 10

dw = 2 * (w*x + b - t) * x
db = 2 * (w*x + b - t)

print("Manual gradients:", dw, db)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

loss = (w * 2 + b - 10) ** 2
loss.backward()

print(f"Autograd: w.grad ={w.grad.item()}, b.grad ={b.grad.item()}")

def train_network(lr):
    rng = np.random.default_rng(0)

    W1 = rng.normal(0, np.sqrt(2/2), (2, 4))
    b1 = np.zeros((1, 4))

    W2 = rng.normal(0, np.sqrt(2/4), (4, 1))
    b2 = np.zeros((1, 1))

    losses = []

    for epoch in range(5000):

        z1 = X @ W1 + b1
        a1 = relu(z1)

        z2 = a1 @ W2 + b2
        output = sigmoid(z2)

        loss = cross_entropy(y, output)
        losses.append(loss)

        dz2 = (output - y) / len(X)

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = (dz2 @ W2.T) * (z1 > 0)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1

        W2 -= lr * dW2
        b2 -= lr * db2

    hidden = relu(X @ W1 + b1)
    predictions = sigmoid(hidden @ W2 + b2)

    return losses, predictions

for lr in [0.01, 0.5, 5]:
    losses, predictions = train_network(lr)

    plt.plot(losses, label=f"Learning Rate = {lr}")

    print("\nLearning Rate:", lr)
    print("Final Loss:", round(losses[-1], 4))
    print("XOR Predictions:", np.round(predictions.flatten(), 2))

plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("XOR Neural Network Training")
plt.legend()
plt.grid()
plt.savefig("outputs/task1_loss.png")
plt.show()