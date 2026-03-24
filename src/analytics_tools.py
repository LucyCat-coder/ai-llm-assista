"""
analytics_tools.py
Модуль статистического и вероятностного анализа для LLM-ассистента психологической поддержки.
Содержит функции для расчёта базовых статистик, условных вероятностей,
байесовского обновления, доверительных интервалов и бутстрэпа.
Все функции сопровождаются комментариями о применении в дипломной работе.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union, Optional

# =============================================================================
# 1. Базовые статистики и работа с выборками (Занятие 1)
# =============================================================================

def mean(values: List[float]) -> float:
    """Среднее арифметическое. Используется для расчёта среднего времени ответа,
    средней длины сообщения, средней оценки пользователя."""
    if len(values) == 0:
        raise ValueError("mean: пустой список")
    return sum(values) / len(values)

def median(values: List[float]) -> float:
    """Медиана. Более устойчива к выбросам, чем среднее. Полезна для оценки
    типичного времени ответа, если бывают очень длинные сессии."""
    if len(values) == 0:
        raise ValueError("median: пустой список")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    else:
        return (sorted_vals[mid-1] + sorted_vals[mid]) / 2.0

def variance_sample(values: List[float]) -> float:
    """Выборочная дисперсия (деление на n-1). Показывает разброс метрик."""
    n = len(values)
    if n < 2:
        raise ValueError("variance_sample: нужно хотя бы 2 элемента")
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (n - 1)

def std_sample(values: List[float]) -> float:
    """Выборочное стандартное отклонение. Мера разброса."""
    return variance_sample(values) ** 0.5

def trimmed_mean(values: List[float], k: int = 1) -> float:
    """Усечённое среднее: отбрасываем k минимальных и k максимальных значений.
    Используется для оценки средней длины запроса без влияния экстремально длинных
    или коротких сообщений (например, спама)."""
    n = len(values)
    if n == 0:
        raise ValueError("trimmed_mean: пустой список")
    if 2 * k >= n:
        raise ValueError("trimmed_mean: слишком большое k")
    sorted_vals = sorted(values)
    core = sorted_vals[k : n-k]
    return mean(core)

def describe(values: List[float]) -> Dict[str, Union[int, float, None]]:
    """Возвращает словарь с основными статистиками выборки.
    Удобно для логирования и формирования отчётов."""
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None,
                "median": None, "std": None}
    n = len(values)
    return {
        "n": n,
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "std": std_sample(values) if n >= 2 else None
    }

# =============================================================================
# 2. Вероятностные функции по частотам (Занятие 3)
# =============================================================================

def prob_event(count_A: int, n: int) -> float:
    """P(A) = count_A / n. Оценка вероятности события по данным."""
    if n <= 0:
        raise ValueError("prob_event: n должно быть > 0")
    if count_A < 0 or count_A > n:
        raise ValueError("prob_event: некорректное count_A")
    return count_A / n

def prob_conditional(count_A_and_B: int, count_B: int) -> float:
    """P(A|B) = count(A∩B) / count(B). Используется, например, для оценки
    вероятности того, что пользователь вернулся после того, как кликнул на совет."""
    if count_B <= 0:
        raise ValueError("prob_conditional: count_B должно быть > 0")
    if count_A_and_B < 0 or count_A_and_B > count_B:
        raise ValueError("prob_conditional: некорректное пересечение")
    return count_A_and_B / count_B

def is_independent_by_counts(p_a: float, p_a_given_b: float, tol: float = 0.05) -> bool:
    """Проверка независимости событий A и B по приближению: |P(A|B)-P(A)| <= tol.
    Помогает понять, влияет ли наличие одного признака на вероятность другого."""
    return abs(p_a_given_b - p_a) <= tol

def contingency_2x2(records: List[Dict], key_a: str, key_b: str) -> List[List[int]]:
    """Строит таблицу сопряжённости 2x2 для двух бинарных признаков (значения 0/1).
    Используется для анализа связи между событиями, например, между срабатыванием
    системы безопасности и реальной опасностью сообщения."""
    table = [[0, 0], [0, 0]]
    for rec in records:
        a = int(rec[key_a])
        b = int(rec[key_b])
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError("contingency_2x2: значения должны быть 0 или 1")
        table[a][b] += 1
    return table

# =============================================================================
# 3. Формула Байеса и сглаживание (Занятие 4)
# =============================================================================

def bayes_posterior(prior: float, likelihood: float, evidence: float) -> float:
    """P(A|B) = (P(B|A) * P(A)) / P(B).
    Используется для динамического обновления вероятности опасности сообщения
    при появлении новых признаков."""
    for name, p in [("prior", prior), ("likelihood", likelihood), ("evidence", evidence)]:
        if p < 0 or p > 1:
            raise ValueError(f"bayes_posterior: {name} должно быть в [0,1]")
    if evidence == 0:
        raise ValueError("bayes_posterior: evidence не может быть 0")
    return (likelihood * prior) / evidence

def laplace_smooth_prob(successes: int, trials: int) -> float:
    """Сглаживание Лапласа: (successes + 1) / (trials + 2).
    Позволяет избежать нулевых вероятностей для редких событий (например,
    для признаков, которые ещё не встречались в опасных сообщениях)."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("laplace_smooth_prob: некорректные аргументы")
    return (successes + 1) / (trials + 2)

# =============================================================================
# 4. Доверительные интервалы и бутстрэп (Занятие 5)
# =============================================================================

def sem(values: List[float]) -> float:
    """Стандартная ошибка среднего: std / sqrt(n).
    Показывает, насколько точно выборочное среднее оценивает истинное."""
    n = len(values)
    if n == 0:
        raise ValueError("sem: пустой список")
    return std_sample(values) / (n ** 0.5)

def ci_mean_normal_approx(values: List[float], z: float = 1.96) -> Tuple[float, float]:
    """Приближённый доверительный интервал для среднего (нормальная аппроксимация):
    mean ± z * SEM. Используется для быстрой оценки неопределённости средней метрики
    (например, средней оценки пользователя)."""
    m = mean(values)
    se = sem(values)
    return (m - z * se, m + z * se)

def bootstrap_means(values: List[float], n_boot: int = 2000, seed: Optional[int] = None) -> List[float]:
    """Генерирует n_boot бутстрэп-средних (выборки с возвращением).
    Основа для построения доверительного интервала методом бутстрэп."""
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    n = len(arr)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = arr[idx]
        means.append(float(np.mean(sample)))
    return means

def bootstrap_ci_mean(values: List[float], n_boot: int = 2000,
                      alpha: float = 0.05, seed: Optional[int] = None) -> Tuple[float, float]:
    """Доверительный интервал для среднего методом бутстрэп (процентильный).
    Более точен, чем нормальная аппроксимация, особенно для малых выборок.
    Возвращает (нижняя_граница, верхняя_граница)."""
    means = bootstrap_means(values, n_boot, seed)
    low = np.quantile(means, alpha / 2)
    high = np.quantile(means, 1 - alpha / 2)
    return (float(low), float(high))

# =============================================================================
# 5. Вспомогательные функции для визуализации (примеры)
# =============================================================================

def plot_hist_with_ci(values: List[float], bootstrap_means: List[float],
                      ci: Tuple[float, float], title: str = "Bootstrap distribution"):
    """Строит гистограмму бутстрэп-средних и отмечает вертикальными линиями
    исходное среднее и границы доверительного интервала."""
    plt.hist(bootstrap_means, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(mean(values), color='red', linestyle='--', label='исходное среднее')
    plt.axvline(ci[0], color='green', linestyle='--', label='нижняя граница CI')
    plt.axvline(ci[1], color='green', linestyle='--', label='верхняя граница CI')
    plt.title(title)
    plt.xlabel('Среднее')
    plt.ylabel('Частота')
    plt.legend()
    plt.show()

def plot_conditional_probs(labels: List[str], probs: List[float], title: str):
    """Столбчатая диаграмма для сравнения условных вероятностей."""
    plt.bar(labels, probs)
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel('Вероятность')
    plt.show()

# =============================================================================
# Пример использования модуля (демонстрация на синтетических данных)
# =============================================================================
if __name__ == "__main__":
    # ----------------------------------------------
    # Имитация данных: логи взаимодействия с ассистентом
    # ----------------------------------------------
    np.random.seed(42)
    n_users = 100

    # Генерируем случайные данные:
    # - response_time: время ответа ассистента (сек)
    # - message_length: длина сообщения пользователя (символы)
    # - user_rating: оценка пользователем ответа (1-5)
    # - clicked_helpful: кликнул ли пользователь на предложенный совет (0/1)
    # - returned_next_day: вернулся ли на следующий день (0/1)
    # - flagged_by_safety: сработала ли система безопасности (0/1)
    # - actually_harmful: было ли сообщение действительно опасным (0/1) – для оценки безопасности

    response_time = np.random.exponential(scale=2.0, size=n_users)  # сек
    message_length = np.random.normal(loc=300, scale=100, size=n_users).clip(10, 1000)
    user_rating = np.random.choice([1,2,3,4,5], size=n_users, p=[0.05,0.1,0.2,0.3,0.35])
    clicked_helpful = np.random.binomial(1, 0.6, size=n_users)
    returned_next_day = np.random.binomial(1, 0.4, size=n_users)
    flagged_by_safety = np.random.binomial(1, 0.1, size=n_users)  # 10% помечено
    actually_harmful = np.random.binomial(1, 0.05, size=n_users)  # 5% реально опасных

    # Собираем в список словарей для вероятностного анализа
    records = []
    for i in range(n_users):
        records.append({
            "clicked": clicked_helpful[i],
            "returned": returned_next_day[i],
            "flagged": flagged_by_safety[i],
            "harmful": actually_harmful[i]
        })

    # ----------------------------------------------
    # 1. Статистический анализ времени ответа
    # ----------------------------------------------
    print("=== Анализ времени ответа ===")
    rt_stats = describe(response_time.tolist())
    print(f"Среднее: {rt_stats['mean']:.2f} сек")
    print(f"Медиана: {rt_stats['median']:.2f} сек")
    print(f"Стд.отклонение: {rt_stats['std']:.2f}")
    print(f"Усечённое среднее (k=2): {trimmed_mean(response_time.tolist(), k=2):.2f} сек")

    # ----------------------------------------------
    # 2. Вероятностный анализ: связь клика и возврата
    # ----------------------------------------------
    print("\n=== Анализ связи клика на совет и возврата ===")
    # Подсчёт частот
    count_clicked = sum(r["clicked"] for r in records)
    count_returned = sum(r["returned"] for r in records)
    count_both = sum(1 for r in records if r["clicked"] == 1 and r["returned"] == 1)

    p_return = prob_event(count_returned, n_users)
    p_return_given_click = prob_conditional(count_both, count_clicked)

    print(f"P(возврат) = {p_return:.3f}")
    print(f"P(возврат | клик) = {p_return_given_click:.3f}")
    independent = is_independent_by_counts(p_return, p_return_given_click, tol=0.05)
    print(f"События независимы? {independent}")

    # Таблица сопряжённости для flagged и harmful
    print("\n=== Таблица сопряжённости: flagged vs harmful ===")
    table = contingency_2x2(records, "flagged", "harmful")
    print("         harmful=0  harmful=1")
    print(f"flagged=0    {table[0][0]}         {table[0][1]}")
    print(f"flagged=1    {table[1][0]}         {table[1][1]}")

    # Оценка качества системы безопасности
    tp = table[1][1]  # flagged=1 и harmful=1
    fp = table[1][0]  # flagged=1 и harmful=0
    fn = table[0][1]  # flagged=0 и harmful=1
    precision = tp / (tp + fp) if (tp+fp)>0 else 0
    recall = tp / (tp + fn) if (tp+fn)>0 else 0
    print(f"Precision (точность) системы безопасности: {precision:.3f}")
    print(f"Recall (полнота) системы безопасности: {recall:.3f}")

    # ----------------------------------------------
    # 3. Байесовское обновление вероятности опасности
    # ----------------------------------------------
    print("\n=== Байесовское обновление ===")
    # Априорная вероятность опасного сообщения P(harmful)
    prior_harmful = prob_event(sum(r["harmful"] for r in records), n_users)
    # Вероятность срабатывания фильтра, если сообщение опасно P(flagged|harmful)
    flagged_given_harmful = prob_conditional(tp, sum(r["harmful"] for r in records))
    # Вероятность срабатывания фильтра вообще P(flagged)
    evidence_flagged = prob_event(sum(r["flagged"] for r in records), n_users)

    posterior = bayes_posterior(prior_harmful, flagged_given_harmful, evidence_flagged)
    print(f"P(harmful) = {prior_harmful:.3f}")
    print(f"P(flagged|harmful) = {flagged_given_harmful:.3f}")
    print(f"P(flagged) = {evidence_flagged:.3f}")
    print(f"P(harmful|flagged) = {posterior:.3f}")

    # Сглаживание Лапласа для признака, который ещё не встречался
    # Например, представим, что у нас нет ни одного опасного сообщения с признаком "содержит слово Х"
    # Тогда без сглаживания P(опасно|признак)=0, а со сглаживанием будет >0.
    smooth_prob = laplace_smooth_prob(successes=0, trials=10)
    print(f"Сглаженная вероятность для признака без наблюдений: {smooth_prob:.3f}")

    # ----------------------------------------------
    # 4. Доверительные интервалы для средней оценки пользователя
    # ----------------------------------------------
    print("\n=== Доверительный интервал для средней оценки ===")
    ratings = user_rating.tolist()
    m_rating = mean(ratings)
    ci_norm = ci_mean_normal_approx(ratings)
    ci_boot = bootstrap_ci_mean(ratings, n_boot=1000, seed=123)

    print(f"Средняя оценка: {m_rating:.2f}")
    print(f"95% CI (норм.аппрокс.): [{ci_norm[0]:.2f}, {ci_norm[1]:.2f}]")
    print(f"95% CI (бутстрэп): [{ci_boot[0]:.2f}, {ci_boot[1]:.2f}]")

    # Визуализация бутстрэп-распределения
    boot_means = bootstrap_means(ratings, n_boot=1000, seed=123)
    plot_hist_with_ci(ratings, boot_means, ci_boot,
                      title="Бутстрэп-распределение средней оценки")

    # ----------------------------------------------
    # 5. Сравнение ширины CI при разном объёме выборки
    # ----------------------------------------------
    print("\n=== Влияние размера выборки на ширину CI ===")
    small_sample = np.random.normal(10, 2, 30).tolist()
    large_sample = np.random.normal(10, 2, 300).tolist()
    ci_small = bootstrap_ci_mean(small_sample, n_boot=1000, seed=1)
    ci_large = bootstrap_ci_mean(large_sample, n_boot=1000, seed=1)
    width_small = ci_small[1] - ci_small[0]
    width_large = ci_large[1] - ci_large[0]
    print(f"Ширина CI для n=30: {width_small:.3f}")
    print(f"Ширина CI для n=300: {width_large:.3f}")
    print("(Чем больше данных, тем уже интервал — оценка точнее)")