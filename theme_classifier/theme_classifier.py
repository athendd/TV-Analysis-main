import torch
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import os
import nltk
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,'../'))
from utils import load_hunterxhunter_single_arc, load_hunterxhunter_series
nltk.download('punkt')
nltk.download('punkt_tab')

class ThemeClassifier():
    
    def __init__(self, theme_list):
        self.model_name = 'facebook/bart-large-mnli'
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)
    
    def load_model(self, device):
        theme_classifier = pipeline(
            'zero-shot-classification',
            model= self.model_name,
            device = device
        )
        
        return theme_classifier
    
    def get_themes_inference(self, script):
        #Split text into sentences
        script_sentences = sent_tokenize(script)
        sentence_batch_size = 25
        script_batches = []
        
        #Go through every 25 sentences to improve runtime
        for index in range(0, len(script_sentences), sentence_batch_size):
            sentence = ' '.join(script_sentences[index:index + sentence_batch_size])
            script_batches.append(sentence)
            
        theme_output = self.theme_classifier(
            script_batches,
            self.theme_list,
            multi_label = True)
        
        themes = {}
        for output in theme_output: 
            for label, score in zip(output['labels'], output['scores']):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)
            
        themes = {key: np.mean(np.array(val)) for key, val in themes.items()}
        
        return themes 
    
    def get_themes(self, dataset_path, save_path = None):
        #Read save output if it exists
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            
            return df, False
                
        dataset_name = dataset_path.split('\\')[-1]
                
        if dataset_name == 'HunterxHunterSubtitles':
            df = load_hunterxhunter_series(dataset_path)
        else:
            df = load_hunterxhunter_single_arc(dataset_path)
        
        output_themes = df['script'].apply(self.get_themes_inference)
        
        themes_df = pd.DataFrame(output_themes.tolist())
        
        df[themes_df.columns] = themes_df
        
        if save_path is not None:
            df.to_csv(save_path, index = False)
                        
        return df, True
