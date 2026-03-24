import matplotlib.pyplot as plt
import numpy as np

# Данные как “события” (список записей)
# Игрушечные данные: 12 пользователей, у каждого (clicked, bought)
records = [
    {"clicked": 1, "bought": 1},
    {"clicked": 1, "bought": 0},
    {"clicked": 1, "bought": 1},
    {"clicked": 0, "bought": 0},
    {"clicked": 1, "bought": 0},
    {"clicked": 1, "bought": 0},
    {"clicked": 1, "bought": 1},
    {"clicked": 0, "bought": 0},
    {"clicked": 0, "bought": 0},
    {"clicked": 1, "bought": 1},
    {"clicked": 1, "bought": 0},
    {"clicked": 1, "bought": 0},
]

print("n =", len(records))
print("first record:", records[0])

#Подсчёт частот (counts) через цикл for

count_clicked = 0
count_bought = 0
count_clicked_and_bought = 0

for r in records:
    if r["clicked"] == 1:
        count_clicked += 1
    if r["bought"] == 1:
        count_bought += 1
    if r["clicked"] == 1 and r["bought"] == 1:
        count_clicked_and_bought += 1

print("count_clicked =", count_clicked)
print("count_bought =", count_bought)
print("count_clicked_and_bought =", count_clicked_and_bought)

# Вероятность события P(A) по частотам

def prob_event(count_A: int, n: int) -> float:
# P(A) = count(A)/n
    if n <= 0:
        raise ValueError("prob_event: n must be > 0")
    if count_A < 0 or count_A > n:
        raise ValueError("prob_event: invalid count")
    return count_A / n

n = len(records)
print("P(clicked) =", prob_event(count_clicked, n))
print("P(bought)  =", prob_event(count_bought, n))

# Условная вероятность P(A|B)

def prob_conditional(count_A_and_B: int, count_B: int) -> float:
# P(A|B) = count(A∩B)/count(B)
    if count_B <= 0:
        raise ValueError("prob_conditional: count_B must be > 0")
    if count_A_and_B < 0 or count_A_and_B > count_B:
        raise ValueError("prob_conditional: invalid intersection count")
    return count_A_and_B / count_B

print("P(bought|clicked) =", prob_conditional(count_clicked_and_bought, count_clicked))

# Проверка независимости по частотам

def is_independent_by_counts(p_a: float, p_a_given_b: float, tol: float = 0.05) -> bool:
# Проверка независимости по приближению |P(A|B)-P(A)| <= tol
    return abs(p_a_given_b - p_a) <= tol
p_bought = prob_event(count_bought, len(records))
p_bought_given_clicked = prob_conditional(count_clicked_and_bought, count_clicked)

print("P(bought) =", round(p_bought, 3))
print("P(bought|clicked) =", round(p_bought_given_clicked, 3))
print("independent? ->", is_independent_by_counts(p_bought, p_bought_given_clicked, tol=0.05))

# Таблица сопряжённости 2×2 (contingency table)
def contingency_2x2(recs: list[dict], a_key: str, b_key: str) -> list[list[int]]:
# 2x2 таблица частот для бинарных признаков a_key и b_key (значения 0/1).
    table = [[0, 0], [0, 0]]
    for r in recs:
        a = int(r[a_key])
        b = int(r[b_key])
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError("contingency_2x2: values must be 0/1")
        table[a][b] += 1
    return table

table = contingency_2x2(records, "clicked", "bought")
print("table 2x2 =", table)

# Визуализация частот (matplotlib)
# доля покупок среди тех, кто НЕ кликнул (clicked=0)
clicked0_total = table[0][0] + table[0][1]
clicked1_total = table[1][0] + table[1][1]

p_buy_given_click0 = table[0][1] / clicked0_total if clicked0_total else 0.0
p_buy_given_click1 = table[1][1] / clicked1_total if clicked1_total else 0.0

plt.bar(["P(buy|click=0)", "P(buy|click=1)"], [p_buy_given_click0, p_buy_given_click1])
plt.ylim(0, 1)
plt.title("Conditional probabilities from data")
plt.show()

print("P(buy|click=0) =", round(p_buy_given_click0, 3))
print("P(buy|click=1) =", round(p_buy_given_click1, 3))

# Монте-Карло: проверяем вероятность симуляцией

def simulate_click_buy(n: int, p_click: float, p_buy_click0: float, p_buy_click1: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    clicked = rng.random(n) < p_click
    bought = np.zeros(n, dtype=bool)
    # для каждого случая выбираем соответствующую вероятность покупки
    probs = np.where(clicked, p_buy_click1, p_buy_click0)
    bought = rng.random(n) < probs
    return clicked, bought

clicked_sim, bought_sim = simulate_click_buy(
    n=100_000,
    p_click=0.75,
    p_buy_click0=0.06,
    p_buy_click1=0.35,
    seed=1
)

# оценим P(buy|click=1) по симуляции
count_click1 = int(clicked_sim.sum())
count_buy_and_click1 = int((bought_sim & clicked_sim).sum())
p_est = count_buy_and_click1 / count_click1

print("simulated P(buy|click=1) ≈", round(p_est, 3))

# Гистограмма оценок вероятности (matplotlib)

def estimate_p_buy_given_click1(n: int, seed: int) -> float:
    clicked_sim, bought_sim = simulate_click_buy(
        n=n, p_click=0.75, p_buy_click0=0.06, p_buy_click1=0.35, seed=seed
    )
    count_click1 = int(clicked_sim.sum())
    count_buy_and_click1 = int((bought_sim & clicked_sim).sum())
    return count_buy_and_click1 / count_click1

estimates = [estimate_p_buy_given_click1(n=20_000, seed=s) for s in range(30)]

plt.hist(estimates, bins=10)
plt.title("Monte Carlo estimates of P(buy|click=1)")
plt.show()

print("mean estimate =", round(float(np.mean(estimates)), 3))