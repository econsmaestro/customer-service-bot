import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

from data import INTENT_LABELS

BASE_MODEL = "bert-base-uncased"


class IntentClassifier:
    """Wraps a BERT sequence classifier fine-tuned on customer-service intents."""

    def __init__(self, model_dir=None):
        if model_dir is not None and model_dir.exists():
            self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
            self.model = BertForSequenceClassification.from_pretrained(model_dir)
        else:
            id2label = {i: label for i, label in enumerate(INTENT_LABELS)}
            label2id = {label: i for i, label in enumerate(INTENT_LABELS)}
            self.tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL)
            self.model = BertForSequenceClassification.from_pretrained(
                BASE_MODEL,
                num_labels=len(INTENT_LABELS),
                id2label=id2label,
                label2id=label2id,
            )
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs))
        return self.model.config.id2label[pred_id], float(probs[pred_id])

    def save(self, model_dir):
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
