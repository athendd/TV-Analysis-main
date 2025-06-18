from glob import glob
import pandas as pd
import os

def load_subtitles_dataset(dataset_path):
    subtitles_paths = glob(dataset_path+'/*.ass')

    scripts=[]
    episode_num=[]
    for path in subtitles_paths:
        #Read Lines
        with open(path,'r', encoding = 'utf-8') as file:
            lines = file.readlines()
            lines = lines[27:]
            lines =  [ ",".join(line.split(',')[9:])  for line in lines ]
        
        lines = [ line.replace('\\N',' ') for line in lines]
        script = " ".join(lines)

        episode = int(path.split('-')[-1].split('.')[0].strip())

        scripts.append(script)
        episode_num.append(episode)
        
    df = pd.DataFrame.from_dict({"episode":episode_num, "script":scripts })
    return df

def load_hunterxhunter_subtitles_dataset(dataset_path):
    scripts_total = []
    episode_num_total = []
    directories = []
    
    for directory in os.listdir(dataset_path):
        if not directory.endswith('.txt'):
            directories.append(directory)
            
    for directory in directories:
        directory_path = dataset_path + f'/{directory}'
        subtitles_path = glob(directory_path + '/*.txt')
        
        scripts = []
        episode_num = []
            
        for path in subtitles_path:
            with open(path, 'r', encoding = 'utf-8') as file:
                lines = file.readlines()
                
            script = ' '.join(lines)
            episode = path.split('\\')[-1].split(' ')[-1].split('.')[0]
            
            scripts.append(script)
            episode_num.append(episode)
        
        scripts_total.extend(script)
        episode_num_total.extend(episode_num)
    
    print(scripts_total)
    print(episode_num_total)
    
#load_hunterxhunter_subtitles_dataset(r'Data\HunterxHunterSubtitles')
load_subtitles_dataset(r'Data\Subtitles')
