import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict, Features, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import nltk
from nltk.corpus import stopwords

# ==================================================
# 1. ЗАГРУЗКА МОДЕЛИ И ДАННЫХ
# ==================================================
model_path = "data/models/emotion_classifier_finetuned"
print("Загрузка модели и токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

with open(os.path.join(model_path, "labels.json"), "r", encoding="utf-8") as f:
    labels_info = json.load(f)
id2label = labels_info["id2label"]
label2id = labels_info["label2id"]
all_labels = [id2label[str(i)] for i in range(len(id2label))]
print("Классы:", all_labels)

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

# Токенизация для валидации
def tokenize_function(examples):
    texts = [t if t is not None else "" for t in examples["text"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

tokenized_val = combined_validation.map(tokenize_function, batched=True)
tokenized_val = tokenized_val.remove_columns(["text"])
tokenized_val = tokenized_val.rename_column("label", "labels")

def convert_labels(example):
    example["labels"] = label2id[example["labels"]]
    return example

tokenized_val = tokenized_val.map(convert_labels)
tokenized_val.set_format("torch")

# Предсказания
trainer = Trainer(model=model)
print("Получение предсказаний на валидационной выборке...")
predictions = trainer.predict(tokenized_val)
preds = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids
probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
max_probs = np.max(probs, axis=1)

# ==================================================
# 2. МАТРИЦА ОШИБОК
# ==================================================
print("\n" + "="*60)
print("АНАЛИЗ 1: МАТРИЦА ОШИБОК")
print("="*60)
print("Матрица ошибок показывает, какие эмоции модель чаще путает.")
print(" - По горизонтали – предсказанные моделью эмоции.")
print(" - По вертикали – истинные эмоции (из разметки).")
print(" - Диагональные элементы – правильно предсказанные примеры (чем больше, тем лучше).")
print(" - Внедиагональные элементы – ошибки. Например, если в строке 'sadness' много попаданий в столбец 'neutral', значит модель не отличает грусть от нейтральности.")
print("\nСтроим график...")

cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=all_labels, yticklabels=all_labels)
plt.xlabel('Предсказанная эмоция', fontsize=12)
plt.ylabel('Истинная эмоция', fontsize=12)
plt.title('Матрица ошибок классификации эмоций', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
print("Закройте окно графика, чтобы продолжить анализ.")

print("\n=== Дополнительная статистика (Classification Report) ===")
print(classification_report(true_labels, preds, target_names=all_labels, zero_division=0))

# ==================================================
# 3. АНАЛИЗ УВЕРЕННОСТИ
# ==================================================
print("\n" + "="*60)
print("АНАЛИЗ 2: УВЕРЕННОСТЬ МОДЕЛИ")
print("="*60)
print("Гистограмма показывает, насколько модель уверена в своих предсказаниях.")
print(" - Ось X: максимальная вероятность (от 0 до 1).")
print(" - Ось Y: количество примеров с такой уверенностью.")
print(" - Чем больше пик справа (высокая вероятность), тем увереннее модель.")
print(" - Красная линия – порог 0.6. Если много примеров слева от неё, модель часто сомневается.")
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
print("Закройте окно графика, чтобы продолжить.")

print(f"\nСтатистика уверенности:")
print(f"  Средняя уверенность: {np.mean(max_probs):.3f}")
print(f"  Медианная уверенность: {np.median(max_probs):.3f}")
print(f"  Доля примеров с уверенностью <0.6: {np.mean(max_probs < 0.6)*100:.1f}%")
if np.mean(max_probs < 0.6) > 0.3:
    print("  ⚠️ Более 30% примеров имеют низкую уверенность. Это сигнал о недостатке данных или плохой разделимости классов.")
else:
    print("  ✅ Большинство предсказаний имеют высокую уверенность.")

# ==================================================
# 4. ТОП-СЛОВА ПО КЛАССАМ
# ==================================================
print("\n" + "="*60)
print("АНАЛИЗ 3: НАИБОЛЕЕ ЧАСТОТНЫЕ СЛОВА ДЛЯ КАЖДОЙ ЭМОЦИИ")
print("="*60)
print("Этот анализ показывает, какие слова чаще всего встречаются в текстах каждой эмоции.")
print("Помогает проверить, соответствуют ли слова ожидаемой эмоции (например, для 'sadness' должны быть 'грустно', 'плакать' и т.д.).")
print("Если для 'positive' встречаются негативные слова – возможно, ошибка в разметке.")
print("\nСначала загрузим стоп-слова и подготовим данные...")

# Подготовка данных для анализа слов
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('russian'))
extra_stop = {'это', 'все', 'так', 'вот', 'там', 'тут', 'который', 'быть', 'сказать', 'знать', 'мочь', 'очень', 'ещё'}
stop_words.update(extra_stop)

# Добавляем числовые метки к train датасету
combined_train = concatenate_datasets([old_dataset["train"], new_split["train"]])
def add_label_id(example):
    example["label_id"] = label2id[example["label"]]
    return example
combined_train = combined_train.map(add_label_id)

def get_top_words(dataset, label_id, top_n=20):
    indices = [i for i, lab in enumerate(dataset["label_id"]) if lab == label_id]
    texts = [dataset["text"][i] for i in indices]
    words = []
    for t in texts:
        if t is None:
            continue
        tokens = [w.lower() for w in str(t).split() if w.lower() not in stop_words and len(w) > 2]
        words.extend(tokens)
    return Counter(words).most_common(top_n)

print("\n=== Топ-15 слов для каждого класса ===")
for label_name, label_id in label2id.items():
    top = get_top_words(combined_train, label_id, top_n=15)
    print(f"\n{label_name}:")
    if not top:
        print("  (нет примеров или недостаточно данных)")
    else:
        for word, count in top:
            print(f"  {word}: {count}")

# Визуализация топ-слов для выбранного класса (sadness)
selected = 'sadness'
if selected in label2id:
    top_words = get_top_words(combined_train, label2id[selected], top_n=10)
    if top_words:
        words, counts = zip(*top_words)
        plt.figure(figsize=(10, 6))
        plt.barh(words, counts, color='skyblue')
        plt.xlabel('Частота', fontsize=12)
        plt.title(f'Топ-10 слов для эмоции "{selected}"', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        print(f"\nГрафик топ-слов для '{selected}' показан. Закройте окно для завершения.")
    else:
        print(f"\nДля класса '{selected}' недостаточно данных для построения графика.")
else:
    print(f"\nКласс '{selected}' не найден в модели.")

print("\n" + "="*60)
print("АНАЛИЗ ЗАВЕРШЁН")
print("="*60)
print("Выводы:")
print("- Если новые классы (sadness, fear, irritation, surprise, disgust, hope) имеют низкую точность (см. classification report) и в матрице ошибок часто попадают в другие классы – необходимо добавить больше примеров для этих эмоций.")
print("- Если уверенность низкая (<0.6) для многих примеров – модель сомневается, нужно больше данных.")
print("- Если топ-слова не соответствуют эмоции – возможно, ошибки в разметке.")