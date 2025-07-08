import os
import torch
import spacy
from transformers import BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer, util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm')


def load_hunterxhunter_single_episode(dataset_path):
    episode_num = int(dataset_path.split("\\")[-1].split(" ")[-1].split(".")[0])
    lines = []
    
    with open(dataset_path, 'r', encoding = 'utf-8') as file:
        lines = file.readlines()
        
        if episode_num >= 1 and episode_num < 22:
            lines = lines[8:]
    
    return lines

def break_into_scenes(script_lines, similarity_threshold=0.7):
    embeddings = embed_model.encode(script_lines, convert_to_tensor=True)
    scenes = []
    current_scene = [script_lines[0]]
    
    for i in range(1, len(script_lines)):
        sim = util.pytorch_cos_sim(embeddings[i-1], embeddings[i]).item()
        if sim < similarity_threshold:
            scenes.append(current_scene)
            current_scene = [script_lines[i]]
        else:
            current_scene.append(script_lines[i])
    
    scenes.append(current_scene)
    
    return scenes

def find_events(scene_lines):
    events = []
    for line in scene_lines:
        doc = nlp(line)
        
        #If line contains at least 1 verb and 1 noun, then its likely an event or action
        has_verb = any(token.pos_ == "VERB" for token in doc)
        has_subject = any(token.pos_ in {"NOUN", "PROPN"} for token in doc)
        if has_verb and has_subject:
            events.append(line)
            
    return events

def summarize_events(scene_lines):
    text = ' '.join(scene_lines)
    inputs = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=1024).to(device)
    
    summary_ids = model.generate(
        inputs,
        max_length=200,
        min_length=80,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

script = load_hunterxhunter_single_episode(r'Data\HunterxHunterSubtitles\Hunter Exam Arc\Episode 2.txt')

#Remove all /n from the script
script = [s.replace('\n', '') for s in script]

scenes = break_into_scenes(script)
events = []

for scene in scenes:
    event_lines = find_events(scene)
    events.extend(event_lines)

if events:
    summary = summarize_events(events)
    print(summary)
else:
    print('no way baby')
    
