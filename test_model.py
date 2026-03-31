import sys
sys.path.append('src/analytic')
from emotion_classifier import EmotionClassifier

model = EmotionClassifier("data/models/emotion_classifier")
texts = [
    "Я боюсь экзамена",
    "Какой отличный день!",
    "Меня всё бесит",
    "Ну да, конечно, гениально!"
    "сижу пишу бота, хочу пить и спать",
    "а если я не хочу наболюдать за дыханием??",
    "не хочу",
    "не люблю практики, мне нравится бегать и орать дурниной как дикий кабан",
    "да че ты заладил одно и то е бесешь",
    "я обиделся",
    "иди нафиг",
    "меня бесит жара",
    "мне грустно"
]
for t in texts:
    print(f"{t} -> {model.predict(t)}")