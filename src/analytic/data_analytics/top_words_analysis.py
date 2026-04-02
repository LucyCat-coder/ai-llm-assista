import os
import json
import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
import matplotlib.pyplot as plt
import math

# Загружаем стоп-слова
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('russian'))
extra_stop = {'это', 'все', 'так', 'вот', 'там', 'тут', 'который', 'быть', 'сказать', 'знать', 'мочь', 'очень', 'ещё'}
stop_words.update(extra_stop)

model_path = "data/models/emotion_classifier_finetuned"
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

combined_train = concatenate_datasets([old_dataset["train"], new_split["train"]])

# Добавляем числовые метки
def add_label_id(example):
    example["label_id"] = label2id[example["label"]]
    return example
combined_train = combined_train.map(add_label_id)

def get_top_words(dataset, label_id, top_n=10):
    indices = [i for i, lab in enumerate(dataset["label_id"]) if lab == label_id]
    texts = [dataset["text"][i] for i in indices]
    words = []
    for t in texts:
        if t is None:
            continue
        tokens = [w.lower() for w in str(t).split() if w.lower() not in stop_words and len(w) > 2]
        words.extend(tokens)
    return Counter(words).most_common(top_n)

# Определяем размер сетки: 3 строки, 4 столбца (11 классов + 1 пустой)
n_classes = len(all_labels)
cols = 4
rows = math.ceil(n_classes / cols)

fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
fig.suptitle('Топ-10 слов для каждой эмоции', fontsize=16)

# Сглаживаем axes для удобного перебора
axes_flat = axes.flatten()

for i, label_name in enumerate(all_labels):
    ax = axes_flat[i]
    label_id = label2id[label_name]
    top_words = get_top_words(combined_train, label_id, top_n=10)
    
    if not top_words:
        ax.text(0.5, 0.5, f'Недостаточно данных\nдля "{label_name}"',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(label_name)
        ax.set_xticks([])
        ax.set_yticks([])
        continue
    
    words, counts = zip(*top_words)
    # Горизонтальная барчарта
    ax.barh(words, counts, color='skyblue')
    ax.set_title(label_name)
    ax.set_xlabel('Частота')
    ax.invert_yaxis()
    # Если слов меньше 10, можно оставить как есть

# Скрываем лишние субплоты (если есть)
for j in range(i+1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # чтобы не перекрывать заголовок
plt.show()
print("График сохранён как top_words_all_classes.png")