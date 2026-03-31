import sys
sys.path.append('src/analytic')
from emotion_classifier import EmotionClassifier

model = EmotionClassifier("data/models/emotion_classifier_finetuned")
texts = [
    "Мне грустно",
    "Всё бесит",
    "Страшно идти одному",
    "Вот это да!",
    "Фу, какая гадость",
    "Надеюсь, всё получится",
    "Какой отличный день!",
    "Я боюсь экзамена"
]
for t in texts:
    print(f"{t} -> {model.predict(t)}")