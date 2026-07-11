import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from data import INTENT_LABELS, REAL_WORLD_TEST_CASES, TRAIN_EXAMPLES
from model import IntentClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "intent_classifier"

EPOCHS = 20
BATCH_SIZE = 8
LR = 2e-5


class IntentDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = {label: i for i, label in enumerate(INTENT_LABELS)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=32, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label2id[label]),
        }


def train():
    classifier = IntentClassifier(model_dir=None)
    dataset = IntentDataset(TRAIN_EXAMPLES, classifier.tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(classifier.model.parameters(), lr=LR)
    classifier.model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = classifier.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
        print(f"epoch {epoch}/{EPOCHS} - loss {total_loss / len(loader):.4f}")

    classifier.model.eval()
    classifier.save(MODEL_DIR)
    print(f"\nSaved fine-tuned model to {MODEL_DIR}")

    print("\nEvaluating on unseen real-world test cases:")
    correct = 0
    for text, expected in REAL_WORLD_TEST_CASES:
        predicted, confidence = classifier.predict(text)
        ok = predicted == expected
        correct += ok
        mark = "OK " if ok else "MISS"
        print(f"  [{mark}] '{text}'\n        expected={expected} predicted={predicted} ({confidence:.2f})")
    print(f"\nReal-world test accuracy: {correct}/{len(REAL_WORLD_TEST_CASES)}")


if __name__ == "__main__":
    train()
