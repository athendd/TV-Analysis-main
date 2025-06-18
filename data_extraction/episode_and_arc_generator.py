import os

def create_arc_folders(arc_name, base_folder_path):
    full_folder_path = base_folder_path + f'/{arc_name}'
    os.makedirs(full_folder_path, exist_ok = True)

def create_episodes(file_path, directory_path):
    try:
        with open(file_path, 'r', encoding = 'utf-8') as file:
            content = file.read()
       
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    separator = "﻿"

    #Split the text by the character
    episodes = content.split(separator)
    episodes.pop(0)
    
    create_episode_files(directory_path, episodes)
    
def create_episode_files(directory_path, episodes):
    for idx in range(0, len(episodes)):
        i = idx + 1
        if i >= 1 and i < 22:
            folder_path = directory_path + f'/Hunter Exam Arc'
            filename = f'Episode {i}.txt'
            
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'w', encoding = 'utf-8') as f:
                f.write(episodes[idx])
        
        elif i >= 22 and i < 27:
            folder_path = directory_path + f'/Zoldyck Family Arc'
            filename = f'Episode {i}.txt'
            
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(episodes[idx])
            
        elif i >= 27 and i < 37:
            folder_path = directory_path + f'/Heavens Arena Arc'
            filename = f'Episode {i}.txt'
            
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'w', encoding = 'utf-8') as f:
                f.write(episodes[idx])
            
        elif i >= 37 and i < 59:
            folder_path = directory_path + f'/Yorknew City Arc'
            filename = f'Episode {i}.txt'
            
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'w', encoding = 'utf-8') as f:
                f.write(episodes[idx])

        elif i >= 59 and i < 76:
            folder_path = directory_path + f'/Greed Island Arc'
            filename = f'Episode {i}.txt'
            
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'w', encoding = 'utf-8') as f:
                f.write(episodes[idx])
            
        elif i >= 76 and i < 137:
            folder_path = directory_path + f'/Chimera Ant Arc'
            filename = f'Episode {i}.txt'
            
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'w', encoding = 'utf-8') as f:
                f.write(episodes[idx])
        
        else:
            folder_path = directory_path + f'/13th Hunter Chairman Election Arc'
            filename = f'Episode {i}.txt'
            
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'w', encoding = 'utf-8') as f:
                f.write(episodes[idx])

hunter_x_hunter_arcs_episodes = {
    "Hunter Exam Arc": "1-21",
    "Zoldyck Family Arc": "22-26",
    "Heavens Arena Arc": "27-36",
    "Yorknew City Arc": "37-58",
    "Greed Island Arc": "59-75",
    "Chimera Ant Arc": "76-136",
    "13th Hunter Chairman Election Arc": "137-148"
}

if __name__ == '__main__':
    arc_keys = list(hunter_x_hunter_arcs_episodes.keys())
    base_folder_path = 'C:/Users/thynnea/Downloads/Personal Projects/TV-Analysis-main/Data/HunterxHunterSubtitles'
    
    for arc_key in arc_keys:
        create_arc_folders(arc_key, base_folder_path)
            
    text_file_path = r'C:\Users\thynnea\Downloads\Personal Projects\TV-Analysis-main\Data\HunterxHunterSubtitles\hunterxhunterdataset copy.txt'
    
    create_episodes(text_file_path, base_folder_path)
        
       
            
            
            
        
            
            
        
    
