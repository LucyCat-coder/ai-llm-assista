import sys
sys.path.append('src/analytic')
from emotion_classifier import EmotionClassifier

model = EmotionClassifier("data/models/emotion_classifier")
texts = [
    "Я боюсь экзамена",
    "Какой отличный день!",
    "Меня всё бесит",
    "Ну да, конечно, гениально!"
]
for t in texts:
    print(f"{t} -> {model.predict(t)}")