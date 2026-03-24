#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Файл запуска анализа данных для дипломной работы.
Загружает логи взаимодействия с LLM-ассистентом,
вычисляет статистики, строит доверительные интервалы и сохраняет результаты.
"""

import pandas as pd
import numpy as np
from src.analytics_tools import (
    describe, bootstrap_ci_mean, prob_conditional,
    contingency_2x2, bayes_posterior, plot_hist_with_ci, bootstrap_means 
)

def load_data(filepath: str):
    """Загружает данные из CSV и преобразует в нужный формат."""
    df = pd.read_csv(filepath)
    # Предположим, что в CSV есть колонки:
    # response_time, user_rating, clicked, returned, flagged, harmful
    return df

def main():
    # 1. Загрузка данных
    print("Загрузка данных...")
    df = load_data("data/logs.csv")
    print(f"Загружено записей: {len(df)}")

    # 2. Базовые статистики для времени ответа
    print("\n=== Анализ времени ответа ===")
    rt_stats = describe(df["response_time"].tolist())
    for key, value in rt_stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

    # 3. Доверительный интервал для средней оценки пользователя
    print("\n=== Доверительный интервал для средней оценки ===")
    ratings = df["user_rating"].tolist()
    ci = bootstrap_ci_mean(ratings, n_boot=2000, seed=42)
    print(f"Средняя оценка: {np.mean(ratings):.2f}")
    print(f"95% бутстрэп-интервал: [{ci[0]:.2f}, {ci[1]:.2f}]")

    # 4. Построение гистограммы бутстрэп-средних (опционально, если нужен график)
    boot_means = bootstrap_means(ratings, n_boot=2000, seed=42)  # функция из модуля
    plot_hist_with_ci(ratings, boot_means, ci, title="Бутстрэп распределение средней оценки")
    # Сохранить график можно через plt.savefig()

    # 5. Анализ системы безопасности с помощью таблицы сопряжённости
    print("\n=== Анализ системы безопасности ===")
    # Преобразуем DataFrame в список словарей для функции contingency_2x2
    records = df[["flagged", "harmful"]].to_dict(orient="records")
    table = contingency_2x2(records, "flagged", "harmful")
    tp, fp, fn = table[1][1], table[1][0], table[0][1]
    precision = tp / (tp + fp) if (tp+fp) > 0 else 0
    recall = tp / (tp + fn) if (tp+fn) > 0 else 0
    print(f"Точность (precision): {precision:.3f}")
    print(f"Полнота (recall): {recall:.3f}")

    # 6. Байесовское обновление вероятности опасности
    print("\n=== Байесовское обновление ===")
    prior = df["harmful"].mean()
    likelihood = tp / df["harmful"].sum() if df["harmful"].sum() > 0 else 0
    evidence = df["flagged"].mean()
    posterior = bayes_posterior(prior, likelihood, evidence)
    print(f"P(harmful) = {prior:.3f}")
    print(f"P(flagged|harmful) = {likelihood:.3f}")
    print(f"P(flagged) = {evidence:.3f}")
    print(f"P(harmful|flagged) = {posterior:.3f}")

    # 7. Сохранение результатов в файл (по желанию)
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("Результаты анализа\n")
        f.write(f"Среднее время ответа: {rt_stats['mean']:.2f}\n")
        f.write(f"Доверительный интервал оценки: [{ci[0]:.2f}, {ci[1]:.2f}]\n")
        f.write(f"Точность системы безопасности: {precision:.3f}\n")
        f.write(f"Полнота системы безопасности: {recall:.3f}\n")

    print("\nАнализ завершён. Результаты сохранены в results.txt")

if __name__ == "__main__":
    main()