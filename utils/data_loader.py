from glob import glob
import pandas as pd
import os
import re   

def episode_cleaner(lines):
    #Checks to see if x appears twice in line both times by itself which is how episode titles for HunterxHunter are spelled
    pattern = r'\b[^x]*\s+x\s+[^x]*\s+x\s+[^x]*\b'
    
    if lines[0].startswith('Fearsome monsters'):
        lines = lines[9:]
    
    cleaned_lines = []
    

    for line in lines:
        line = line.strip()
        match = re.search(pattern, line)
        
        if "Gon and Killua's Hunterpedia" in line or "G.I. Tutorial" in line:
            break
        
        if not line.startswith('Next time:') and not '[' in line and not match:
            cleaned_lines.append(line)
            
    return cleaned_lines

def load_hunterxhunter_arc_subtitles(directory, directory_path, scripts, episode_num):
    subtitles_path = glob(directory_path + '/*.txt')
                    
    for path in subtitles_path:
        with open(path, 'r', encoding = 'utf-8') as file:
            lines = file.readlines()
        
        #Remove the first 8 lines from file since its just narration     
        if directory == 'Hunter Exam Arc':
            lines = lines[8:]  
            
        cleaned_lines = episode_cleaner(lines)
        
        script = '\n '.join(cleaned_lines)
        episode = int(path.split('\\')[-1].split(' ')[-1].split('.')[0])

        scripts.append(script)
        episode_num.append(episode)
        
def load_hunterxhunter_single_episode(dataset_path):
    episode_num = int(dataset_path.split("\\")[-1].split(" ")[-1].split(".")[0])
    lines = []
    
    with open(dataset_path, 'r', encoding = 'utf-8') as file:
        lines = file.readlines()
        
        if episode_num >= 1 and episode_num < 22:
            lines = lines[8:]
        
    cleaned_lines = episode_cleaner(lines)
                
    return '\n'.join(cleaned_lines)
        
def load_hunterxhunter_single_arc(dataset_path):
    scripts = []
    episode_num = []
    directory = dataset_path.split("\\")[-1]
    
    load_hunterxhunter_arc_subtitles(directory, dataset_path, scripts, episode_num)
    
    df = pd.DataFrame.from_dict({"episode":episode_num, "script":scripts })
    
    return df

def load_hunterxhunter_series(dataset_path):
    scripts = []
    episode_num = []
    directories = []
    
    #Get all arc directories
    for directory in os.listdir(dataset_path):
        if not directory.endswith('.txt'):
            directories.append(directory)
                
    for directory in directories:
        directory_path = dataset_path + f'/{directory}'
        
        load_hunterxhunter_arc_subtitles(directory, directory_path, scripts, episode_num)
    
    df = pd.DataFrame.from_dict({"episode":episode_num, "script":scripts })
    
    return df
