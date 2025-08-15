import json
import os
import tensorflow_datasets as tfds
import torch
from transformers import pipeline
from typing import List, Dict, Union

class QuoteAnalyzer:
    def __init__(self):
        self.tone_model = "facebook/bart-large-mnli"  
        self.device = 0 if torch.cuda.is_available() else -1
        self.tone_labels = ["teasing", "cocky", "sincere", "protective", "sarcastic", "playful", "threatening"]
        self.tone_pipe = None
        self.emotion_classifer = None
        self.updated_quotes = []  

    def process_file(self, file_path):
        data = self._load_data(file_path)
        if data:
            quotes = data.get("Quotes", [])

            if quotes and isinstance(quotes, list):
                quotes = self.clean_quotes(quotes)
                self.updated_quotes = self._classify_tones_and_emotions(quotes)
                data['Quotes'] = self.updated_quotes
                self._update_data(file_path, data)

    def setup_tone_model(self):
        self.tone_pipe = pipeline(
            "zero-shot-classification",
            model=self.tone_model,
            device=self.device
        )
        
    def setup_emotion_model(self):
        self.emotion_classifer = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=1)

    def _classify_tones_and_emotions(self, quotes):
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
        emotion_results = self._classify_emotions(texts)
                
        out = []
        for i in range(len(texts)):
            out.append({
                "Quote": texts[i],
                "Emotion": emotion_results[i],
                "Tone": tone_results[i]["labels"][0],
            })
            
        print(out)
            
        return out
    
    def _classify_tones(self, texts):
        self.setup_tone_model()
        results = self.tone_pipe(
            texts,
            candidate_labels=self.tone_labels,
            hypothesis_template="The speaker's tone is {}.",
            multi_label=False
        )
        
        if isinstance(results, dict):
            results = [results]
            
        return results
    
    def _classify_emotions(self, texts):
        self.setup_emotion_model()
        emotions = []
        for text in texts:
            emotions.append(self.emotion_classifer(text)[0][0]['label'])
        
        return emotions        

    def _load_data(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
            
        return None
    
    def _update_data(self, file_path, data):
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def get_emotions_data(self):
        ds, _ = tfds.load('goemotions', split=['train', 'test', 'validation'], with_info=True)
        
        return ds[0], ds[1], ds[2]
    
    @staticmethod
    def clean_quotes(quotes):
        fixed_quotes = []
        for quote in quotes:
            fixed_quotes.append(quote.replace('\"', ''))
            
        return fixed_quotes

qa = QuoteAnalyzer()
qa.process_file(r'Data\Character_Data_for_Chatbot\Hisoka_Morow')


