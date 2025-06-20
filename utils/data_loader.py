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
        break
    
    df = pd.DataFrame.from_dict({"episode":episode_num, "script":scripts })
    return df

def load_hunterxhunter_arc_subtitles(directory, directory_path, scripts, episode_num):
    subtitles_path = glob(directory_path + '/*.txt')
                    
    for path in subtitles_path:
        with open(path, 'r', encoding = 'utf-8') as file:
            lines = file.readlines()
        
        #Remove the first 8 lines from file since its just narration     
        if directory == 'Hunter Exam Arc':
            lines = lines[8:]  
        
        script = ' '.join(lines)
        episode = int(path.split('\\')[-1].split(' ')[-1].split('.')[0])

        scripts.append(script)
        episode_num.append(episode)
        
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