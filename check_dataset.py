import pandas as pd

# Путь к вашему CSV с новыми примерами
csv_path = "data/raw/new_emotions.csv"

df = pd.read_csv(csv_path, encoding="utf-8")
print(f"Всего строк: {len(df)}")
print("\nРаспределение по классам:")
print(df['label'].value_counts())