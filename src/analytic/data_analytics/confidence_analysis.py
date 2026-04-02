import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

model_path = "data/models/emotion_classifier_finetuned"

print("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

with open(os.path.join(model_path, "labels.json"), "r", encoding="utf-8") as f:
    labels_info = json.load(f)
label2id = labels_info["label2id"]

# Данные (аналогично предыдущему скрипту)
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
print("Получение предсказаний...")
predictions = trainer.predict(tokenized)
probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
max_probs = np.max(probs, axis=1)

print("\n" + "="*60)
print("АНАЛИЗ УВЕРЕННОСТИ МОДЕЛИ")
print("="*60)
print("Гистограмма показывает, насколько модель уверена в своих предсказаниях.")
print("Ось X: максимальная вероятность (от 0 до 1).")
print("Ось Y: количество примеров с такой уверенностью.")
print("Чем больше пик справа (высокая вероятность), тем увереннее модель.")
print("Красная линия – порог 0.6. Если много примеров слева от неё, модель часто сомневается.")
print("\nСтроим график...")

plt.figure(figsize=(10, 6))
sns.histplot(max_probs, bins=50, color='green', kde=True)
plt.axvline(x=0.6, color='red', linestyle='--', label='Порог уверенности 0.6')
plt.xlabel('Максимальная вероятность предсказания', fontsize=12)
plt.ylabel('Количество примеров', fontsize=12)
plt.title('Распределение уверенности модели', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nСтатистика уверенности:")
print(f"  Средняя уверенность: {np.mean(max_probs):.3f}")
print(f"  Медианная уверенность: {np.median(max_probs):.3f}")
print(f"  Доля примеров с уверенностью <0.6: {np.mean(max_probs < 0.6)*100:.1f}%")
if np.mean(max_probs < 0.6) > 0.3:
    print("  ⚠️ Более 30% примеров имеют низкую уверенность. Нужно больше данных.")
else:
    print("  ✅ Большинство предсказаний имеют высокую уверенность.")