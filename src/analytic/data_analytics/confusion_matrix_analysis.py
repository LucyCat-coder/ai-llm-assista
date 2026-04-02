import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import confusion_matrix, classification_report

model_path = "data/models/emotion_classifier_finetuned"
print("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

with open(os.path.join(model_path, "labels.json"), "r", encoding="utf-8") as f:
    labels_info = json.load(f)
id2label = labels_info["id2label"]
label2id = labels_info["label2id"]
all_labels = [id2label[str(i)] for i in range(len(id2label))]

print("Загрузка датасетов...")
old_dataset = load_dataset("Kostya165/ru_emotion_dvach")
new_df = pd.read_csv("data/raw/new_emotions.csv", encoding="utf-8")
new_dataset = Dataset.from_pandas(new_df)
new_split = new_dataset.train_test_split(test_size=0.2, seed=42)

old_text_type = old_dataset["train"].features["text"]
old_label_type = old_dataset["train"].features["label"]
new_split["train"] = new_split["train"].cast(Features({"text": old_text_type, "label": old_label_type}))
new_split["test"] = new_split["test"].cast(Features({"text": old_text_type, "label": old_label_type}))

combined_validation = concatenate_datasets([old_dataset["validation"], new_split["test"]])

def tokenize_function(examples):
    texts = [t if t is not None else "" for t in examples["text"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

tokenized = combined_validation.map(tokenize_function, batched=True)
tokenized = tokenized.remove_columns(["text"])
tokenized = tokenized.rename_column("label", "labels")

def convert_labels(example):
    example["labels"] = label2id[example["labels"]]
    return example
tokenized = tokenized.map(convert_labels)
tokenized.set_format("torch")

trainer = Trainer(model=model)
predictions = trainer.predict(tokenized)
preds = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

print("\n" + "="*60)
print("МАТРИЦА ОШИБОК")
print("="*60)
print("По горизонтали – предсказанные эмоции, по вертикали – истинные.")
print("Диагональ – правильные ответы. Чем ярче и больше число на диагонали, тем лучше.")
print("Если вне диагонали есть большие числа – модель путает соответствующие классы.")
print("Например, если sadness часто попадает в neutral, значит модель не различает грусть и нейтральность.")
print("\nСтроим график...")

cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=all_labels, yticklabels=all_labels)
plt.xlabel('Предсказанная эмоция')
plt.ylabel('Истинная эмоция')
plt.title('Матрица ошибок классификации эмоций')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nClassification report (метрики по каждому классу):")
print(classification_report(true_labels, preds, target_names=all_labels, zero_division=0))