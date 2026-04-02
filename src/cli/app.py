# src/cli/app.py
import sys
import os

# Добавляем корень проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytic.emotional_rag_pipeline import EmotionRAGPipeline

def main():
    model_path = "data/models/emotion_classifier_finetuned"
    kb_path = "data/knowledge_base/emotion_advice.csv"

    pipeline = EmotionRAGPipeline(model_path, kb_path)
    print("Чат-бот готов. Введите текст (или 'exit' для выхода):")
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == "exit":
            break
        response, emotion = pipeline.process(user_input)
        print(f"Бот: {response}")

if __name__ == "__main__":
    main()