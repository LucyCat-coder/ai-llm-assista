import numpy as np
import matplotlib.pyplot as plt 

# создайте rng с seed=42 и сгенерируйте data: normal(loc=10, scale=2, size=50)
# raise NotImplementedError("TODO: generate data")

rng = np.random.default_rng(42)
data = rng.normal(loc=10.0, scale=2.0, size=100)  # 50 наблюдений
len(data), float(data[0])

# Функция mean (без pandas)

def mean(values) -> float:
    # Среднее арифметическое для списка/массива
    # проверьте пустой список и верните sum(values)/len(values)
    
    if len(values) == 0:
        raise ValueError("mean: empty")
    return float(sum(values)) / len(values)
    
m = mean(data)
m

# Выборочное std: std_sample

# дисперсия (sample variance) делится на n-1
# стандартное отклонение = sqrt(variance)
# генератор: sum((x-m)**2 for x in values)

def variance_sample(values) -> float:
    # TODO: реализуйте выборочную дисперсию: sum((x-m)^2)/(n-1)
    # raise NotImplementedError("TODO: implement variance_sample")
    n = len(values)
    if n < 2:
        raise ValueError("variance_sample: need >=2")
    m = mean(values)
    return float(sum((x - m) ** 2 for x in values)) / (n - 1)

def std_sample(values) -> float:
    # TODO: верните sqrt(variance_sample(values))
    # raise NotImplementedError("TODO: implement std_sample")
    return variance_sample(values) ** 0.5
s = std_sample(data)
s

# Стандартная ошибка среднего (SEM) SEM = std / sqrt(n)
def sem(values) -> float:
    # реализуйте SEM = std_sample(values)/sqrt(n)
    if n <= 0:
        raise ValueError("sem: empty")
    return std_sample(values) / (n ** 0.5)
    
sem_val = sem(data)
sem_val

# Приближённый 95% CI для среднего (normal approx) I95 ≈ mean ± 1.96 * SEM
# Это приближение. Для небольших n точнее использовать t-распределение, но для этого курса на данном этапе достаточно приближения + бутстрэпа.

def ci_mean_normal_approx(values, z: float = 1.96):
    # Приближённый CI для среднего: mean ± z*SEM.
    # посчитайте m=mean(values), se=sem(values) и верните (m - z*se, m + z*se)

    se = sem(values)
    return (m - z * se, m + z * se)
    
ci_norm = ci_mean_normal_approx(data)
ci_norm

# Бутстрэп: “новая выборка из старой” с возвращением

# Выбираем n элементов из данных с возвращением. Считаем среднее.Повторяем много раз

#rng.integers(0, n, size=n) — индексы для ресэмплинга

def bootstrap_means(values, n_boot: int = 2000, seed: int = 0):
    # реализуйте список из n_boot средних, полученных бутстрэпом
    
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        raise ValueError("bootstrap_means: empty")
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # выборка индексов с возвращением
        sample_b = values[idx]
        means.append(float(np.mean(sample_b)))
    return means
    
boot = bootstrap_means(data, n_boot=2000, seed=1)
len(boot), boot[0]

# Bootstrap CI через квантили
# Для 95% CI берём нижний квантиль 2.5% верхний квантиль 97.5%
# np.quantile(array, q)

def bootstrap_ci_mean(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    # получите means=bootstrap_means(...), верните квантили (alpha/2, 1-alpha/2)
    
    means = bootstrap_means(values, n_boot=n_boot, seed=seed)
    low = float(np.quantile(means, alpha/2))
    high = float(np.quantile(means, 1 - alpha/2))
    return low, high 
    
ci_boot = bootstrap_ci_mean(data, n_boot=2000, alpha=0.05, seed=1)
ci_boot

# График: распределение бутстрэп-средних + линии CI (matplotlib)

# гистограмма бутстрэп-средних, вертикальные линии: mean, CI_low, CI_high

# постройте гистограмму boot (из ячейки 6) и нанесите 3 вертикальные линии: mean(data), ci_boot[0], ci_boot[1]

boot_means = boot  # из ячейки 6
ci_low, ci_high = ci_boot
m_hat = mean(data)

plt.hist(boot_means, bins=30)
plt.axvline(m_hat, linestyle="--")
plt.axvline(ci_low, linestyle="--")
plt.axvline(ci_high, linestyle="--")
plt.title("Bootstrap means + 95% CI (lines)")
plt.show()

print("mean =", round(m_hat, 3))
print("bootstrap CI =", (round(ci_low, 3), round(ci_high, 3)))

# Эксперимент: как размер выборки влияет на ширину CI (matplotlib)
# создайте data20 и data200, посчитайте ci20 и ci200, ширины width20/width200 и постройте bar chart

rng = np.random.default_rng(7)

data30 = rng.normal(10.0, 3.0, 30)
data300 = rng.normal(10.0, 3.0, 300)

ci30 = bootstrap_ci_mean(data30, n_boot=1500, seed=2)
ci300 = bootstrap_ci_mean(data300, n_boot=1500, seed=2)

width30 = ci30[1] - ci30[0]
width300 = ci300[1] - ci300[0]

plt.bar(["n=30", "n=300"], [width30, width300])
plt.title("CI width shrinks when n grows")
plt.show()

print("width n=30  =", round(width30, 3))
print("width n=300 =", round(width300, 3))