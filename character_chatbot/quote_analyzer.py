import logging
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

class QuoteAnalyzer:
    def __init__(self):
        self.tone_model = "facebook/bart-large-mnli"
        self.device = 0 if torch.cuda.is_available() else -1
        self.tone_labels = ["teasing", "cocky", "sincere", "protective", "sarcastic", "playful", "threatening"]

        self.tone_pipe = None
        self.emotion_classifier = None

    def process_quotes(self, quotes):
        """quotes: list[str] or list[{'Quote': str}]"""
        texts = []
        for q in quotes:
            if isinstance(q, dict):
                texts.append(str(q.get("Quote", "")).strip())
            else:
                texts.append(str(q).strip())

        texts = [t for t in texts if t]
        if not texts:
            return []

        tone_results = self._classify_tones(texts)
        emotion_labels = self._classify_emotions(texts)

        out = []
        for i in range(len(texts)):
            out.append({
                "Quote": texts[i],
                "Emotion": emotion_labels[i],
                "Tone": tone_results[i]["labels"][0]
            })
        return out

    def setup_tone_model(self):
        if self.tone_pipe is None:
            logger.info("Loading tone model: %s", self.tone_model)
            self.tone_pipe = pipeline(
                "zero-shot-classification",
                model=self.tone_model,
                device=self.device
            )

    def setup_emotion_model(self):
        if self.emotion_classifier is None:
            model_id = "SamLowe/roberta-base-go_emotions"
            logger.info("Loading emotion model: %s", model_id)
            self.emotion_classifier = pipeline(
                task="text-classification",
                model=model_id,
                top_k=1,
                device=self.device
            )

    def _classify_tones(self, texts):
        self.setup_tone_model()
        results = self.tone_pipe(
            texts,
            candidate_labels=self.tone_labels,
            hypothesis_template="The speaker's tone is {}.",
            multi_label=False,
            truncation=True,
            batch_size=16,
        )
        if isinstance(results, dict):
            results = [results]

        return results

    def _classify_emotions(self, texts):
        self.setup_emotion_model()
        output = self.emotion_classifier(texts, top_k=1, truncation=True, batch_size=32)
        labels = []
        for item in output:
            labels.append(item[0]["label"])

        return labels