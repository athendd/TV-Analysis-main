import requests
from bs4 import BeautifulSoup
import requests
import json 
import re
import os

class CharacterDataExtractor:
    
    def __init__(self, character_name):
        self.character_name = character_name
        page = requests.get(f"https://hunterxhunter.fandom.com/wiki/{self.character_name}")
                
        if page.status_code != 200:
            raise ValueError("Unable to retrieve character data")
        
        self.soup = BeautifulSoup(page.content, 'html.parser')
        self.character_dict = {}
        self.character_dict['Name'] = self.character_name.replace('_', ' ')
        
    def save_dict(self):
        folder_path = r'Data\Character_Data_for_Chatbot'
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        full_file_path = os.path.join(folder_path, self.character_name)
        

        with open(full_file_path, 'w') as f:
            json.dump(self.character_dict, f, indent=4)
        
    def create_and_save_character(self):
        section_titles = ['Appearance', 'Personality', 'Background', 'Plot', 'Equipment', 'Abilities & Powers', 'Quotes']
        self.get_aside()
        
        for section_title in section_titles:
            self.get_section_paragraph(section_title)        
            
        self.save_dict()
        
    def get_aside(self):
        aside_object = self.soup.find('aside')
        
        if aside_object != None:
            self.character_dict['Physical Features'] = {}
            physical_features = ['hair', 'eyes', 'height', 'weight', 'blood', 'gender']
            
            for physical_feature in physical_features:
                self.character_dict['Physical Features'][physical_feature.capitalize()] = self.find_data_source(aside_object, physical_feature)
                
            other_traits = ['affiliation', 'previous affiliation', 'status', 'birthday', 'age', 'type', 'relatives']
            for other_trait in other_traits:
                if other_trait == 'age':
                    self.character_dict[other_trait.capitalize()] = self.get_age(self.find_data_source(aside_object, other_trait))
                else:
                    self.character_dict[other_trait.capitalize()] = self.find_data_source(aside_object, other_trait) 
                                    
    def find_data_source(self, aside_object, data_source_name):
        element = aside_object.find(attrs={"data-source": data_source_name})
        if element != None:
            if data_source_name == 'relatives':
                val = self.soup.find("div", {"data-source": "relatives"}).find(class_="pi-data-value")

                relatives = {}

                for a in val.find_all("a"):
                    name = a.get_text(strip=True)
                    bits = []
                    for sib in a.next_siblings:
                        if getattr(sib, "name", None) == "br":
                            break
                        bits.append(sib.get_text(" ", strip=True) if hasattr(sib, "get_text") else str(sib))
                    tail = " ".join(bits).strip()

                    m = re.search(r"\(([^)]+)\)", tail)
                    if not m:
                        continue 
                    relationship = m.group(1).strip().replace('"', "")

                    if relationship.lower() == 'status unknown' or relationship.lower() == 'unknown relation':
                        continue

                    relatives[relationship] = name
                    
                return relatives
            
            else:
                if data_source_name in ['affiliation', 'previous affiliation']:
                    a_eles = element.find_all('a')
                    all_eles = []
                    for a_ele in a_eles:
                        text = self.clean_text(a_ele.get_text())
                        if text != '':
                            all_eles.append(text)
                    
                    return ', '.join(all_eles)
                else:
                    
                    text = element.get_text()
                    words = text.split()
                    remaining_words = None
                    if data_source_name in ['blood', 'eyes', 'hair']:
                        remaining_words = words[2:]
                    else:
                        remaining_words = words[1:]
                    
                    new_word = ' '.join(remaining_words)
                    if data_source_name in ['birthday', 'height', 'weight', 'hair', 'eyes']:
                        new_word = new_word.split('(')[0]
                        
                    element_text = new_word.replace('\n', '')
                    
                    return self.clean_text(element_text)
        
        return None
        
    def get_section_paragraph(self, section_title):
        headers = self.soup.find_all(['h2'])
        
        for header in headers:
            header_name = header.get_text()
            
            if header_name == section_title:
                if section_title == 'Plot':
                    self.plot_arc_split(header)
                elif section_title == 'Abilities & Powers':
                    self.get_power_and_abilities(header)
                elif section_title == 'Quotes':
                    self.get_quotes(header)
                elif section_title == 'Equipment':
                    self.get_equipment(header)
                else:
                    self.character_dict[section_title] = ''
                    for sibling in header.find_next_siblings():
                        if sibling.name in ['h2']:
                            break                            
                        if sibling.name == 'p':
                            self.character_dict[section_title] += self.clean_text(sibling.get_text())
                                            
    def get_power_and_abilities(self, header):
        self.character_dict['Abilities & Powers'] = {}
        self.character_dict['Abilities & Powers']['Description'] = ''
        nen = False
        for sibling in header.find_next_siblings():
            if sibling.name in ['h2']:
                break
            if sibling.name in ['h3']:
                nen = True
                self.character_dict['Abilities & Powers']['Nen'] = {}
                self.character_dict['Abilities & Powers']['Nen']['Description'] = ''

            if sibling.name == 'p':
                p_text = self.clean_text(sibling.get_text())

                if nen:
                    self.character_dict['Abilities & Powers']['Nen']['Description'] += self.clean_text(p_text)
                first_sentence = p_text.split('.')[0]
                if ':' in first_sentence:
                    self.character_dict['Abilities & Powers'][first_sentence.split(':')[0].strip()] = self.clean_text(p_text.split(':')[1].strip())
                else:
                    self.character_dict['Abilities & Powers']['Description'] += self.clean_text(p_text)
                    
            if sibling.name == 'table':
                self.get_nen_abilites(sibling)
                    
    def get_nen_abilites(self, table):
        rows = table.find_all("tr")
        for idx, row in enumerate(rows):
            if idx == 0:
                continue
            ths = row.find_all("th")
            if len(ths) >= 2:
                nen_type = self.clean_text(ths[0].get_text().split(':')[1])
                nen_name = self.clean_text(ths[1].get_text())
                
                if not nen_name or not nen_type:
                    continue
                
                description = None
                
                for sibling in row.find_next_siblings('tr'):
                    if len(sibling.find_all("th")) >= 2:
                        break

                    tds = sibling.find_all("td")
                    if not tds:
                        continue
                    best = ""
                    

                    for td in tds:
                        if td.find("figure"):
                            continue
                        txt = self.clean_text(td.get_text())
                        if len(txt) > len(best):
                            
                            best = txt
                                                    
                    if best:
                        description = best
                        break
                    
                if description:
                    self.character_dict['Abilities & Powers']['Nen'][nen_name] = {}
                    self.character_dict['Abilities & Powers']['Nen'][nen_name]['Description'] = description
                    self.character_dict['Abilities & Powers']['Nen'][nen_name]['Type'] = nen_type
                            
    def get_quotes(self, header):
        self.character_dict['Quotes'] = []
        for sibling in header.find_next_siblings():
            if sibling.name in ['h2']:
                break
            if sibling.name == 'div':
                ul = sibling.find("ul")  
                self.character_dict['Quotes'].extend([self.clean_text(' '.join(li.get_text(strip=True).split())) for li in ul.find_all("li")])
            
    def get_equipment(self, header):
        self.character_dict['Equipment'] = {}
        
        for sibling in header.find_next_siblings():
            if sibling.name in ['h2']:
                break
            if sibling.name == 'p':
                equipment_name = sibling.get_text().split(':')[0].strip()
                equipment_description = sibling.get_text().split(':')[1].strip()
                self.character_dict['Equipment'][equipment_name] = self.clean_text(equipment_description)
        
    def plot_arc_split(self, header):
        self.character_dict['Plot'] = {}
        current_arc = ''
        for sibling in header.find_next_siblings():
            if sibling.name in ['h2']:
                break
            if sibling.name in ['h3']:
                current_arc = sibling.get_text()
                self.character_dict['Plot'][current_arc] = ''
                
            if sibling.name == 'p' and current_arc != '':
                self.character_dict['Plot'][current_arc] += self.clean_text(sibling.get_text())
                
    @staticmethod
    def clean_text(text):
        if text == None:
            return None
        
        text = text.replace('*', '')
        
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\[[^\]]*\]', '', text)
            
        return text.strip()
    
    @staticmethod
    def get_age(text):
        if text == None:
            return None
        numbers = re.findall(r'\d+', text)
        if numbers:
            return numbers[-1]
        else:
            return None
                        
character_data_extractor = CharacterDataExtractor('Gon_Freecss')
character_data_extractor.create_and_save_character()
    
        
    
