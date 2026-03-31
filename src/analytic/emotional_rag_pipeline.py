# src/analytic/emotional_rag_pipeline.py
from .emotion_classifier import EmotionClassifier
from .knowledge_base import KnowledgeBase

class EmotionRAGPipeline:
    def __init__(self, classifier_path, kb_path):
        self.classifier = EmotionClassifier(classifier_path)
        self.kb = KnowledgeBase(kb_path)

    def process(self, text, randomize_advice=True):
        emotion = self.classifier.predict(text)
        advice = self.kb.get_advice(emotion, randomize=randomize_advice)
        response = f"Я понимаю, что вы чувствуете {emotion}. {advice}"
        return response, emotion