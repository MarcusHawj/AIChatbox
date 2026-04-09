"""
train.py — Run this ONCE before launching the app.
It reads intents.json, trains the neural network, and saves trained_model.pth.

Usage:
  python train.py
"""

import json
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ── Simple stemmer (no external libraries needed) ─────────────────────────────
class SimpleStemmer:
    SUFFIXES = [
        ("ational", "ate"), ("tional", "tion"), ("izing", "ize"),
        ("ising", "ise"), ("ness", ""), ("ment", ""), ("ful", ""),
        ("ous", ""), ("ive", ""), ("able", ""), ("ible", ""),
        ("ant", ""), ("ent", ""), ("ism", ""), ("ate", ""),
        ("al", ""), ("er", ""), ("ic", ""), ("ly", ""),
        ("ed", ""), ("ing", ""),
    ]

    def stem(self, word):
        word = word.lower()
        for suffix in ["sses", "ies"]:
            if word.endswith(suffix):
                return word[:-2]
        if word.endswith("ss") or word.endswith("us"):
            return word
        if word.endswith("s") and len(word) > 3:
            word = word[:-1]
        for suffix, replacement in self.SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)] + replacement
        return word


stemmer = SimpleStemmer()


def tokenize(sentence):
    return re.findall(r"\b\w+\b", sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokens, all_words):
    stemmed = [stem(w) for w in tokens]
    return np.array([1.0 if w in stemmed else 0.0 for w in all_words], dtype=np.float32)


# ── Neural Network ─────────────────────────────────────────────────────────────
class ChatNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        return self.net(x)


# ── Load intents ───────────────────────────────────────────────────────────────
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

all_words, tags, xy = [], [], []
IGNORE = set("?!.,'-")

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        words = tokenize(pattern)
        all_words.extend(words)
        xy.append((words, tag))

all_words = sorted(set(stem(w) for w in all_words if w not in IGNORE))
tags = sorted(set(tags))
print(f"✔  {len(tags)} intents | {len(all_words)} unique words | {len(xy)} training samples")

# ── Build training data ────────────────────────────────────────────────────────
X = np.array([bag_of_words(words, all_words) for words, _ in xy])
y = np.array([tags.index(tag) for _, tag in xy])


class ChatDataset(Dataset):
    def __len__(self): return len(X)
    def __getitem__(self, i): return torch.from_numpy(X[i]), torch.tensor(y[i])


loader = DataLoader(ChatDataset(), batch_size=8, shuffle=True)

# ── Train ──────────────────────────────────────────────────────────────────────
HIDDEN = 256
EPOCHS = 1500
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChatNet(len(all_words), HIDDEN, len(tags)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"\n🚀 Training on {device} ...\n")
for epoch in range(EPOCHS):
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item()
    if (epoch + 1) % 250 == 0:
        print(f"  Epoch {epoch+1:>5}/{EPOCHS}  |  Loss: {total/len(loader):.5f}")

print("\n✅ Training complete!")

torch.save({
    "model_state": model.state_dict(),
    "input_size": len(all_words),
    "hidden_size": HIDDEN,
    "output_size": len(tags),
    "all_words": all_words,
    "tags": tags,
}, "trained_model.pth")

print("💾 Saved → trained_model.pth")
print("\nNow run:  python app.py")
