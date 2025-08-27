from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .util_methods import load_dict  
import os
from typing import List, Dict, Any, Optional


SECTION_PLOT = 'plot'
SECTION_APPEARANCE = 'appearance'
SECTION_BACKGROUND = 'background'
SECTION_ABILITY_NEN = 'ability_nen'
SECTION_ABILITY_OTHER = 'ability_other'
SECTION_EQUIPMENT = 'equipment'
SECTION_PROFILE = 'profile'


"""
Produces documents full of character data for each character from HunterxHunter
"""
class CharacterRetriever:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.characters_dicts = self._get_all_characters_data(self.folder_path)

    """
    Creates all documents for each character

    Returns:
        -(list): A list of documents for each character
    """
    def create_character_documents(self):
        if not self.characters_dicts:
            return []
        documents: List[Document] = []
        for character_dict in self.characters_dicts:
            docs = self._create_documents_for_character(character_dict)
            if docs:
                documents.extend(docs)
        if documents:
            print(documents[0].page_content[:200])
            print(documents[0].metadata)

        return documents

    """
    Builds a map for each character which is used for instant lookups without retrieval

    Returns:
        -(dict): With the key name for the character's name which has its own dictionary containing the character's
        persona card, voice cues, status, birthday, and nen type
    """
    def build_profile_map(self):
        if not self.characters_dicts:
            return {}
        
        profiles = {}
        for ch in self.characters_dicts:
            name = (ch.get('name') or '').strip()
            if not name:
                continue

            profiles[name] = {
                'persona_card': ch.get('persona_card', ''),
                'voice_cues': ch.get('voice_cues', ''),
                'status': ch.get('status', ''),
                'birthday': ch.get('birthday', ''),
                'nen_type': ch.get('nen_type', '')
            }

        return profiles

    """
    Creates documents for a single character by extracting data from the given character dictionary and saving it to the document

    Args:
        -character_dict (dict): A dictionary full of data on the current character
    Returns:
        -(list): A list of documents that represent the current character
    """
    def _create_documents_for_character(self, character_dict):
        docs = []

        name = character_dict.get('name', '').strip()
        if not name:
            return None

        source_tag = f'hxh tag for {name}'

        profile_blob = self._get_persona_and_voice_cues(character_dict)
        if profile_blob:
            docs.append(
                Document(
                    page_content=profile_blob,
                    metadata={'name': name, 'section': SECTION_PROFILE, 'source_tag': source_tag},
                )
            )

        for section in (SECTION_APPEARANCE, SECTION_BACKGROUND):
            raw = character_dict.get(section, '') or ''
            docs.extend(self._chunk_to_docs(raw, name, section, source_tag))

        nen_text = self._nen_abilities_to_text(character_dict.get('nen_abilities', {}) or {})
        docs.extend(self._chunk_to_docs(nen_text, name, SECTION_ABILITY_NEN, source_tag))

        other_text = self._kv_dict_to_sentences(character_dict.get('non-nen_abilities', {}) or {})
        docs.extend(self._chunk_to_docs(other_text, name, SECTION_ABILITY_OTHER, source_tag))

        equip_text = self._kv_dict_to_sentences(character_dict.get('equipment', {}) or {})
        docs.extend(self._chunk_to_docs(equip_text, name, SECTION_EQUIPMENT, source_tag))

        plot_dict = character_dict.get('plot', {}) or {}
        for arc_name, arc_text in plot_dict.items():
            for i, chunk in enumerate(self._split_text(arc_text), start=1):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            'name': name,
                            'section': SECTION_PLOT,
                            'arc': arc_name,
                            'part': i,
                            'source_tag': source_tag,
                        },
                    )
                )

        return docs

    """
    Obtains the persona and voices cues text from the given character dictionary

    Args:
        -character_dict (dict): A dictionary full of data on the current character
    Returns:
        -(str): A combination of persona and voices cues data
    """
    @staticmethod
    def _get_persona_and_voice_cues(character_dict):
        persona = (character_dict.get('persona_card') or '').strip()
        voice = (character_dict.get('voice_cues') or '').strip()
        if not (persona or voice):
            return None
        
        lines = []
        if persona:
            lines.append(f'Persona: {persona}')
        if voice:
            lines.append(f'Voice Cues: {voice}')

        return '\n'.join(lines).strip()

    """
    Converts a given dictionary to a single piece of text

    Args:
        -dictionary (dict): The given dictionary
    Returns:
        -(str): A string containing the key and value from all key value pairs in the given dictionary
    """
    @staticmethod
    def _kv_dict_to_sentences(dictionary):
        parts = []
        for k, v in dictionary.items():
            if v is None:
                continue

            parts.append(f'{k}: {v}.')

        return ' '.join(parts).strip()

    """
    Converts the nen abilites dictionary into one string

    Args:
        -nen_abilities (dict): Where each key is the name of a nen ability and the value is the ability's desrcription
    Returns:
        -(str): A string of all the nen abilities 
    """
    @staticmethod
    def _nen_abilities_to_text(nen_abilities):
        parts = []
        for key, val in nen_abilities.items():
            desc = (val.get('Description') or '').strip()
            typ = (val.get('Type') or '').strip()
            if desc and typ:
                parts.append(f'{key}: {desc} (Nen Type: {typ}).')
            elif desc:
                parts.append(f'{key}: {desc}.')
            elif typ:
                parts.append(f'{key}: (Nen Type: {typ}).')

        return ' '.join(parts).strip()

    """
    Convert chunks of text into a list of documents

    Args:
        -text (str): A large section of text
        -name (str): Name of the character
        -section (str): Section of character data
        -source_tag: Source tag that will be used to identify the document
    Returns:
        -(list): List of documents
    """
    def _chunk_to_docs(self, text, name, section, source_tag):
        text = (text or '').strip()
        if not text:
            return []
        
        chunks = self._split_text(text)
        out: List[Document] = []
        for i, chunk in enumerate(chunks, start=1):
            md = {'name': name, 'section': section, 'source_tag': source_tag}
            if len(chunks) > 1:
                md['part'] = i
            out.append(Document(page_content=chunk, metadata=md))

        return out

    """
    Split the given text into chunks

    Args:
        -text (str): A piece of text
        -chunk_size (int): The size of each chunk in terms of characters
        -chunk_overlap (int): The overlap between each chunk in terms of characters
    Returns:
        -(list): A list of pieces of the given text
    """
    def _split_text(self, text, chunk_size = 500, chunk_overlap = 60):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = len
        )

        return [t.strip() for t in splitter.split_text(text or '') if t and t.strip()]

    """
    Retrieve all of the characters dictionaries inside the given folder

    Args:
        -folder_path (str): A string of the relative locatin of the folder containing all the characters data
    Returns:
        -(list): A list of characters dictionary with each dictionary representing a single character
    """
    @staticmethod
    def _get_all_characters_data(folder_path):
        out = []
        try:
            for file_name in os.listdir(folder_path):
                if not file_name.endswith('.json'):
                    continue
                if not file_name[:-5].endswith('Updated'):
                    continue
                full_path = os.path.join(folder_path, file_name)
                curr = load_dict(full_path)
                if curr:
                    out.append(curr)

            return out
        except FileNotFoundError:
            print(f'Error: Directory {folder_path} not found')
            return []
        except Exception as e:
            print(f'An error occurred: {e}')
            return []