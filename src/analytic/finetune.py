import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset, DatasetDict, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

torch.set_num_threads(2)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

def main():
    print("=== Начало дообучения ===")

    # Путь к исходной модели
    model_path = "data/models/emotion_classifier"
    output_dir = "data/models/emotion_classifier_finetuned"

    # 1. Загружаем старый датасет
    print("Загрузка старого датасета...")
    old_dataset = load_dataset("Kostya165/ru_emotion_dvach")
    old_text_type = old_dataset["train"].features["text"]
    old_label_type = old_dataset["train"].features["label"]
    print(f"Типы в старом датасете: text={old_text_type}, label={old_label_type}")

    # 2. Загружаем новый CSV
    csv_path = "data/raw/new_emotions.csv"
    if not os.path.exists(csv_path):
        print(f"Файл {csv_path} не найден! Создайте его с примерами.")
        return
    print(f"Загрузка нового датасета из {csv_path}...")
    new_df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"Добавлено {len(new_df)} новых примеров.")
    new_dataset = Dataset.from_pandas(new_df)

    # 3. Разделяем новый датасет на train/validation
    new_split = new_dataset.train_test_split(test_size=0.2, seed=42)

    # 4. Приводим типы нового датасета к типам старого
    print("Приведение типов нового датасета...")
    new_split["train"] = new_split["train"].cast(Features({
        "text": old_text_type,
        "label": old_label_type
    }))
    new_split["test"] = new_split["test"].cast(Features({
        "text": old_text_type,
        "label": old_label_type
    }))

    # 5. Объединяем датасеты
    print("Объединение датасетов...")
    combined_train = concatenate_datasets([old_dataset["train"], new_split["train"]])
    combined_validation = concatenate_datasets([old_dataset["validation"], new_split["test"]])

    combined = DatasetDict({
        "train": combined_train,
        "validation": combined_validation
    })

    # 6. Определяем все классы
    all_labels = sorted(set(combined["train"]["label"]))
    # Удаляем возможный артефакт 'label', если он есть
    if 'label' in all_labels:
        all_labels.remove('label')
    print("Все классы:", all_labels)
    id2label = {i: label for i, label in enumerate(all_labels)}
    label2id = {label: i for i, label in enumerate(all_labels)}

    # 7. Токенизация
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize_function(examples):
        texts = [t if t is not None else "" for t in examples["text"]]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

    tokenized = combined.map(tokenize_function, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")

    def convert_labels(example):
        example["labels"] = label2id[example["labels"]]
        return example

    tokenized = tokenized.map(convert_labels)
    tokenized.set_format("torch")

    # 8. Загружаем модель
    print("Загрузка модели...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # разрешить несовпадение размеров классификатора
    )

    # 9. Аргументы обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    print("Начинаем дообучение...")
    trainer.train()

    # 10. Сохраняем модель
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    labels_info = {"id2label": id2label, "label2id": label2id}
    with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels_info, f, ensure_ascii=False, indent=2)

    print(f"Дообучение завершено. Модель сохранена в {output_dir}")

if __name__ == "__main__":
    main()