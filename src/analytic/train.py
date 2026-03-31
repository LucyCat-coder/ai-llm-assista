import os
import sys
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

# Ограничиваем потоки PyTorch, чтобы снизить нагрев
torch.set_num_threads(2)

# Добавляем корень проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

def main():
    # ========== ВСЕ НАСТРОЙКИ ЗДЕСЬ ==========
    dataset_name = "Kostya165/ru_emotion_dvach"
    model_name = "cointegrated/rubert-tiny2"
    output_dir = "data/models/emotion_classifier"
    os.makedirs(output_dir, exist_ok=True)

    # Параметры обучения (без YAML, сразу числами)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,      # снижено для стабильности
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=False,                         # отключаем, т.к. нет GPU
    )
    # =======================================

    print("Загрузка датасета...")
    dataset = load_dataset(dataset_name)

    # Опционально: для быстрого теста (раскомментируйте, если нужно)
    # dataset["train"] = dataset["train"].select(range(500))
    # if "validation" in dataset:
    #     dataset["validation"] = dataset["validation"].select(range(100))

    labels_list = sorted(set(dataset["train"]["label"]))
    id2label = {i: label for i, label in enumerate(labels_list)}
    label2id = {label: i for i, label in enumerate(labels_list)}
    print(f"Метки: {labels_list}")

    print("Токенизация...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        texts = [t if t is not None else "" for t in examples["text"]]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")

    def convert_labels(example):
        example["labels"] = label2id[example["labels"]]
        return example

    tokenized = tokenized.map(convert_labels)
    tokenized.set_format("torch")

    print("Загрузка модели...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels_list),
        id2label=id2label,
        label2id=label2id,
    )

    # Определяем валидационный сплит
    eval_split = None
    for split in ['validation', 'valid']:
        if split in tokenized:
            eval_split = split
            break
    if eval_split is None:
        raise ValueError("Нет валидационного сплита. Доступны: " + ", ".join(tokenized.keys()))
    print(f"Используем валидационный сплит: {eval_split}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        compute_metrics=compute_metrics,
    )

    print("Начинаем обучение...")
    trainer.train()

    # Сохраняем метки и токенизатор
    labels_info = {"id2label": id2label, "label2id": label2id}
    with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels_info, f, ensure_ascii=False, indent=2)
    tokenizer.save_pretrained(output_dir)
    print(f"Модель сохранена в {output_dir}")

if __name__ == "__main__":
    main()