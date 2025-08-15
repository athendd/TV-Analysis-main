from textattack.augmentation import WordNetAugmenter
import pandas as pd
from queue import PriorityQueue
from collections import defaultdict
import nltk
from utils import load_nen_data
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class DataAugmenter():
    
    def __init__(self, filepath, save_path, num_augments = 900):
        self.filepath = filepath
        self.save_path = save_path
        self.num_augments = num_augments
        self.df = None
        self.augmenter = WordNetAugmenter()
        
    def augment_data(self):
        try:
            data = load_nen_data(self.filepath)
        except Exception as e:
            print('Unable to load data')
        
        if data == None or data == {}:
            return None
        
        self.df = self.convert_dictlist_to_dataframe(data)
        
        user_abilities_map = self.create_user_abilities_map()
        user_queue, user_freq = self.create_user_queue_and_freq()
        nen_dict = self.create_nen_type_freq()
        
        for _ in range(self.num_augments):        
            _, user = user_queue.get()

            #Get user's abilities
            abilities = user_abilities_map[user]

            #Find the ability with least-represented Nen types
            chosen_ability = self.get_chosen_ability(abilities, nen_dict)
            if chosen_ability is None:
                continue  
                    
            new_sentence = self.create_augmented_text(chosen_ability)
            if new_sentence == []:
                continue
                    
            aug_text = ' '.join(new_sentence)
            new_sample = {
                'Name': chosen_ability['Name'] + '_aug',
                'Description': aug_text,
                'Character_Name': user,
                'Types': chosen_ability['Types']
            }
            
            self.df.loc[len(self.df)] = new_sample
                    
            user_freq[user] += 1
            for nen_type in chosen_ability['Types']:
                nen_dict[nen_type] += 1

            user_queue.put((user_freq[user], user))

        self.save_augmented_data()
    
    def create_augmented_text(self, chosen_ability):
        base_text = chosen_ability['Description']
        sentences = sent_tokenize(base_text)
        new_sentence = []
        for sentence in sentences:
            new_sentence.append(self.augmenter.augment(sentence)[0])
            
        return new_sentence
            
    def get_chosen_ability(self, abilities, nen_dict):
        chosen_ability = None
        chosen_score = float('inf')
        for ability_name in abilities:
            ability_row = self.get_ability_row(df, ability_name)
            types = ability_row['Types']
            score = min(nen_dict[t] for t in types)
            if score < chosen_score:
                chosen_ability = ability_row
                chosen_score = score
                
        return chosen_ability
             
    def create_user_abilities_map(self):
        user_dict = defaultdict(list)
        for index, row in self.df.iterrows():
            user_dict[row['Character_Name']].append(row['Name'])
            
        return user_dict
    
    def create_user_queue_and_freq(self):
        users = self.df['Character_Name'].unique().tolist()
        user_freq = self.df['Character_Name'].value_counts().to_dict()
        user_queue = PriorityQueue()
        for user in users:
            user_queue.put((user_freq[user], user))
            
        return user_queue, user_freq
    
    def create_nen_type_freq(self):
        nen_dict = defaultdict(int)
        for _, row in self.df.iterrows():
            for nen_type in row['Types']:
                nen_dict[nen_type] += 1
                
        return nen_dict
    
    def get_ability_row(self, ability_name):
        return self.df[self.df['Name'] == ability_name].iloc[0]
    
    def save_augmented_data(self):
        self.df.to_json(self.save_path, orient='records', lines=True)
                    
    @staticmethod
    def convert_dictlist_to_dataframe(dict_list):
        return pd.DataFrame.from_dict(dict_list)