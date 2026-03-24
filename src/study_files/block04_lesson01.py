sample = [23, 57, 84, 12, 91, 45, 68, 33, 76, 49, 54, 92, 32, 61, 73]

'''среднее: sum, len, функция mean'''

def mean(values: list[float]) -> float:
    """Среднее арифметическое. Требует непустой список."""
    # TODO: проверьте пустой список и верните sum(values)/len(values)
    if len(values) == 0:
      raise NotImplementedError("TODO: implement mean")
    return sum(values) / len(values)
print("mean =", mean(sample))

'''медиана: sorted, индексы, чёт/нечёт'''

def median(values: list[float]) -> float:
    """Медиана. Требует непустой список."""
    # TODO: реализуйте медиану через сортировку и проверку чётности n
    if len(values) == 0:
      raise NotImplementedError("TODO: implement median")
    sort = sorted(values)
    num = len(sort)
    mid = num // 2
    if num % 2 == 1:
      return float(sort[mid])
    else:
      return (sort[mid - 1] + sort[mid]) / 2
      
print("median =", median(sample))

'''выборочная дисперсия: сумма квадратов отклонений'''

def variance_sample(values: list[float]) -> float:
    """Выборочная дисперсия (деление на n-1)."""
    # TODO: проверьте n>=2, найдите m=mean(values), верните sum((x-m)**2)/(n-1)
    n = len(values)
    if n < 2:
      raise NotImplementedError("TODO: implement variance_sample")
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (n - 1)

print("variance_sample =", variance_sample(sample))

'''стандартное отклонение: корень из дисперсии'''

def std_sample(values: list[float]) -> float:
    """Выборочное стандартное отклонение."""
    # TODO: верните (variance_sample(values))**0.5
    return variance_sample(values) ** 0.5
    raise NotImplementedError("TODO: implement std_sample")

print("std_sample =", std_sample(sample))

'''выбросы: сравним mean и median “до/после”'''

def with_outlier(values: list[float], outlier: float) -> list[float]:
    """Вернуть новую выборку, добавив выброс (не меняем исходный список)."""
    # TODO: верните новый список list(values) + [outlier]
    return list(values) + [outlier]
    raise NotImplementedError("TODO: implement with_outlier")

sample_out = with_outlier(sample, 100)

print("mean(before) =", round(mean(sample), 3), "median(before) =", median(sample))
print("mean(after)  =", round(mean(sample_out), 3), "median(after)  =", median(sample_out))

'''trimmed mean: устойчивое среднее'''

def trimmed_mean(values: list[float], k: int = 1) -> float:
    """Усечённое среднее: убрать k минимальных и k максимальных."""
    # TODO: проверьте n>0 и 2*k<n, затем core=sorted(values)[k:n-k], return mean(core)
    n = len(values)
    if n == 0:
      raise NotImplementedError("TODO: implement trimmed_mean")
    if 2 * k >= n:
      raise NotImplementedError("TODO: implement trimmed_mean")
    s = sorted(values)
    core = s[k:n - k]
    return mean(core)
    
print("trimmed_mean(before) =", round(trimmed_mean(sample, k=1), 3))
print("trimmed_mean(after)  =", round(trimmed_mean(sample_out, k=1), 3))