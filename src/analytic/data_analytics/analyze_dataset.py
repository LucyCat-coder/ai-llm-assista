import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Настройка стиля
sns.set_style("whitegrid")

# 1. Загружаем старый датасет
print("Загрузка старого датасета...")
old_dataset = load_dataset("Kostya165/ru_emotion_dvach")
old_labels = old_dataset["train"]["label"]
old_counts = pd.Series(old_labels).value_counts().sort_index()

# 2. Загружаем новый CSV
print("Загрузка нового датасета...")
new_df = pd.read_csv("data/raw/new_emotions.csv", encoding="utf-8")
new_counts = new_df['label'].value_counts().sort_index()

# 3. Объединённое распределение
combined_counts = old_counts.add(new_counts, fill_value=0).astype(int).sort_index()

# 4. График распределения классов (три подграфика)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
old_counts.plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Старый датасет (5 классов)')
axes[0].set_xlabel('Эмоция')
axes[0].set_ylabel('Количество')

new_counts.plot(kind='bar', ax=axes[1], color='salmon')
axes[1].set_title('Новый датасет (добавленные примеры)')
axes[1].set_xlabel('Эмоция')

combined_counts.plot(kind='bar', ax=axes[2], color='lightgreen')
axes[2].set_title('Объединённый датасет')
axes[2].set_xlabel('Эмоция')

plt.tight_layout()
plt.show()  # Открывает окно с графиком

# 5. Анализ длины текстов
old_texts = old_dataset["train"]["text"]
new_texts = new_df['text'].tolist()
all_texts = list(old_texts) + new_texts
lens_chars = [len(str(t)) for t in all_texts if t is not None]

plt.figure(figsize=(10, 4))
sns.histplot(lens_chars, bins=50, color='purple')
plt.title('Распределение длины текстов (в символах)')
plt.xlabel('Длина')
plt.ylabel('Частота')
plt.axvline(x=128, color='red', linestyle='--', label='max_length = 128')
plt.legend()
plt.show()  # Открывает второе окно

# 6. Печать статистики в консоль
print("\n=== Статистика по классам ===")
print("Старый датасет:")
print(old_counts)
print(f"\nНовый датасет:")
print(new_counts)
print(f"\nОбъединённый:")
print(combined_counts)

print(f"\nСредняя длина текста: {sum(lens_chars)/len(lens_chars):.1f} символов")
print(f"Медианная длина: {pd.Series(lens_chars).median():.1f}")
print(f"Максимальная длина: {max(lens_chars)}")

# 7. Примеры для проверки качества разметки
print("\n=== Проверка примеров (первые 2 из нового датасета для каждого класса) ===")
for label in combined_counts.index:
    examples = new_df[new_df['label'] == label]['text'].head(2).tolist()
    if examples:
        print(f"\n{label}:")
        for ex in examples:
            print(f"  - {ex[:100]}")