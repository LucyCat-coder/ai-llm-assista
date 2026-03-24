import pandas as pd
import numpy as np

np.random.seed(42)
n = 200  # количество записей

data = {
    "response_time": np.random.exponential(scale=2.0, size=n).round(2),   # время ответа
    "user_rating": np.random.choice([1,2,3,4,5], size=n, p=[0.05,0.1,0.2,0.3,0.35]),
    "clicked": np.random.binomial(1, 0.6, size=n),                        # кликнул на совет
    "returned": np.random.binomial(1, 0.4, size=n),                       # вернулся на следующий день
    "flagged": np.random.binomial(1, 0.1, size=n),                        # сработала система безопасности
    "harmful": np.random.binomial(1, 0.05, size=n)                        # реально опасное сообщение
}

df = pd.DataFrame(data)
df.to_csv("data/logs.csv", index=False)
print("Тестовые данные созданы в data/logs.csv")