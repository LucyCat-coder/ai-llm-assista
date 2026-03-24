records = [
    {"clicked": 1, "bought": 1},
    {"clicked": 0, "bought": 0},
    {"clicked": 1, "bought": 1},
    {"clicked": 0, "bought": 0},
    {"clicked": 1, "bought": 0},
    {"clicked": 0, "bought": 0},
    {"clicked": 1, "bought": 0},
    {"clicked": 0, "bought": 0},
    {"clicked": 1, "bought": 0},
    {"clicked": 0, "bought": 1},
    {"clicked": 0, "bought": 0},
    {"clicked": 1, "bought": 0},
    {"clicked": 1, "bought": 1},
    {"clicked": 0, "bought": 0},
    {"clicked": 1, "bought": 1},
    {"clicked": 1, "bought": 0},
    {"clicked": 1, "bought": 0},
    {"clicked": 0, "bought": 1},
    {"clicked": 1, "bought": 0},
    {"clicked": 0, "bought": 0},
    {"clicked": 1, "bought": 1},
    {"clicked": 1, "bought": 1},
    {"clicked": 1, "bought": 0},
    {"clicked": 1, "bought": 0},
]
print("n =", len(records))

# Счётчики (частоты) для Байеса
def build_binary_counts(recs: list[dict], a_key: str, b_key: str) -> dict:
    # посчитайте n, count_A, count_B, count_A_and_B и верните словарь

    n = len(recs)
    count_A = 0
    count_B = 0
    count_A_and_B = 0

    for rec in recs:
      a = int(rec[a_key])
      b = int(rec[b_key])
      if a not in (0, 1) or b not in (0, 1):
        raise ValueError(f"Invalid value for {a_key}: {a}")
      if a == 1:
        count_A += 1
      if b == 1:
        count_B += 1
      if a == 1 and b == 1:
        count_A_and_B += 1
    return {"n": n, "count_A": count_A, "count_B": count_B, "count_A_and_B": count_A_and_B}

counts = build_binary_counts(records, "bought", "clicked")
counts

# P = count / n (функция prob_from_counts) 
# Безопасно считаем вероятность из частоты.

def prob_from_counts(count: int, n: int) -> float:
    #  проверьте n>0, корректность count и верните count/n
    
    
    if n <= 0:
        raise ValueError("prob_from_counts: n must be > 0")
    if count < 0 or count > n:
        raise ValueError("prob_from_counts: invalid count")
    return count / n
    
print("P(bought)  =", prob_from_counts(counts["count_A"], counts["n"]))
print("P(clicked) =", prob_from_counts(counts["count_B"], counts["n"]))

# Likelihood P(B|A) по данным
# P(clicked | bought) = count(A∩B) / count(A)

def prob_conditional(count_A_and_B: int, count_A: int) -> float:
    # реализуйте безопасно и верните count_A_and_B/count_A
    
    if count_A <= 0:
        raise ValueError("prob_conditional: condition count must be > 0")
    if count_A_and_B < 0 or count_A_and_B > count_A:
        raise ValueError("prob_conditional: invalid intersection count")
    return count_A_and_B / count_A

p_B_given_A = prob_conditional(counts["count_A_and_B"], counts["count_A"])
print("P(clicked|bought) =", p_B_given_A)

# Формула Байеса (bayes_posterior)
# P(bought|clicked) = P(clicked|bought)*P(bought)/P(clicked)

def bayes_posterior(prior: float, likelihood: float, evidence: float) -> float:
    # проверьте значения в [0,1], evidence>0, верните (likelihood*prior)/evidence

    for name, p in [("prior", prior), ("likelihood", likelihood), ("evidence", evidence)]:
        if p < 0 or p > 1:
            raise ValueError(f"bayes_posterior: {name} must be in [0,1]")
    if evidence == 0:
        raise ValueError("bayes_posterior: evidence must be > 0")
    return (likelihood * prior) / evidence

prior = prob_from_counts(counts["count_A"], counts["n"])
evidence = prob_from_counts(counts["count_B"], counts["n"])
likelihood = p_B_given_A

posterior = bayes_posterior(prior, likelihood, evidence)
print("P(bought|clicked) via Bayes =", posterior)

# Проверка: прямой подсчёт vs Байес
# Прямо из данных: P(bought|clicked) = count(A∩B) / count(B)

direct = counts["count_A_and_B"] / counts["count_B"]
print("direct =", direct)
print("bayes  =", posterior)
print("diff   =", abs(direct - posterior))


# Наивный скоринг: P(buy | clicked=value)
# Если clicked=1 → используем P(buy|click=1)
# Если clicked=0 → используем P(buy|click=0)

def score_buy_probability(recs: list[dict], clicked_value: int) -> float:
    #  отфильтруйте subset по clicked_value, посчитайте bought_count и верните bought_count/len(subset)
    
    if clicked_value not in (0, 1):
        raise ValueError("clicked_value must be 0/1")
    subset = [r for r in recs if int(r["clicked"]) == clicked_value]
    if len(subset) == 0:
        raise ValueError("No records for clicked_value")
    bought_count = sum(1 for r in subset if int(r["bought"]) == 1)
    return bought_count / len(subset)

p_buy_click1 = score_buy_probability(records, 1)
p_buy_click0 = score_buy_probability(records, 0)

print("P(buy|click=1) =", p_buy_click1)
print("P(buy|click=0) =", p_buy_click0)

# Сглаживание Лапласа (Laplace smoothing)
# Чтобы избежать нулевых вероятностей: P = (successes + 1) / (trials + 2)

def laplace_smooth_prob(successes: int, trials: int) -> float:
    # TODO: проверьте корректность и верните (successes+1)/(trials+2)
    # raise NotImplementedError("TODO: implement laplace_smooth_prob")

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("laplace_smooth_prob: invalid counts")
    return (successes + 1) / (trials + 2)

# посчитайте сглаженную вероятность для clicked=0

subset0 = [r for r in records if int(r["clicked"]) == 0]
succ0 = sum(1 for r in subset0 if int(r["bought"]) == 1)
p_smooth0 = laplace_smooth_prob(succ0, len(subset0))

print("raw P(buy|click=0) =", p_buy_click0)
print("smooth P(buy|click=0) =", p_smooth0)