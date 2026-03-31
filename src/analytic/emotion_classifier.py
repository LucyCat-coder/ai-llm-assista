"""
emotion_classifier.py
Класс для загрузки и использования обученной модели классификации эмоций.
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        # Загружаем метки
        with open(os.path.join(model_path, "labels.json"), "r", encoding="utf-8") as f:
            self.labels_info = json.load(f)
        self.id2label = self.labels_info["id2label"]

    def predict(self, text):
        """
        Принимает строку текста, возвращает предсказанную эмоцию (строка).
        """
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            pred_id = torch.argmax(logits, dim=-1).item()
        return self.id2label[str(pred_id)]