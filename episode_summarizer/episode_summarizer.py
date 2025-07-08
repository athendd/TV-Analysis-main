import torch
import spacy
from transformers import BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer, util
import os
from utils import load_hunterxhunter_single_episode

nlp = spacy.load('en_core_web_sm')

class EpisodeSummarizer():
    
    def __init__(self, similarity_threshold = 0.6):
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(self.device)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.similarity_threshold = similarity_threshold
    
    def break_into_scenes(self, script_lines):
        embeddings = self.embedder.encode(script_lines, convert_to_tensor=True)
        scenes = []
        current_scene = [script_lines[0]]
        
        for i in range(1, len(script_lines)):
            sim = util.pytorch_cos_sim(embeddings[i-1], embeddings[i]).item()
            if sim < self.similarity_threshold:
                scenes.append(current_scene)
                current_scene = [script_lines[i]]
            else:
                current_scene.append(script_lines[i])
        
        scenes.append(current_scene)
        
        return scenes
    
    def find_events(self, scene_lines):
        events = []
        for line in scene_lines:
            doc = nlp(line)
            
            #If line contains at least 1 verb and 1 noun, then its likely an event or action
            has_verb = any(token.pos_ == "VERB" for token in doc)
            has_subject = any(token.pos_ in {"NOUN", "PROPN"} for token in doc)
            if has_verb and has_subject:
                events.append(line)
                
        return events
    
    def summarize_events(self, scene_lines):
        text = ' '.join(scene_lines)
        inputs = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=1024).to(self.device)
        
        summary_ids = self.model.generate(
            inputs,
            max_length=200,
            min_length=80,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
    def get_episode_summary(self, data_path, save_path = None):
        summary = ''

        if os.path.exists(save_path):
            with open(save_path, 'r', encoding = 'utf-8') as file:
                summary = file.read()

            return summary
        
        script = load_hunterxhunter_single_episode(data_path)

        #Remove all /n from the script
        script = [s.replace('\n', '') for s in script]

        scenes = self.break_into_scenes(script)
        events = []

        for scene in scenes:
            event_lines = self.find_events(scene)
            events.extend(event_lines)

        if events:
            summary = self.summarize_events(events)
            
            if save_path is not None:
                with open(save_path, 'w') as file:
                    file.write(summary)
        else:
            summary = 'Unable to extract summarization from text'
            
        return summary
