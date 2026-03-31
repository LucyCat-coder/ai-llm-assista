"""
knowledge_base.py
Класс для загрузки базы знаний (эмоция -> совет) из CSV.
"""

import pandas as pd
import random

class KnowledgeBase:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, encoding="utf-8")
        # Группируем советы по эмоциям
        self.advice_by_emotion = self.df.groupby("emotion")["advice"].apply(list).to_dict()

    def get_advice(self, emotion, randomize=True):
        """
        Возвращает совет для заданной эмоции.
        Если randomize=True, возвращает случайный совет из списка.
        Иначе — первый.
        """
        if emotion in self.advice_by_emotion:
            advice_list = self.advice_by_emotion[emotion]
            if randomize:
                return random.choice(advice_list)
            else:
                return advice_list[0]
        else:
            # Эмоция не найдена в базе
            return "Постарайтесь сделать глубокий вдох и обратиться к специалисту, если чувство сохраняется."