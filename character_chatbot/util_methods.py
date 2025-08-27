import os 
import json

def load_dict(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data_dict = json.load(file)
            
            return data_dict
    except Exception as e:
        print('Unable to obtain data dictionary')
        return None

def convert_quotes_to_sentences(quotes):
    sentences = []
    for quote in quotes:
        quote_text = quote.get('Quote', '').strip()
        quote_tone = quote.get('Tone', '')
        quote_emotion = quote.get('Emotion', '')
        if quote_text:
            sentences.append(f"'{quote_text}' (Emotion: {quote_emotion}, Tone: {quote_tone})")

    text_start = 'QUOTES (with emotion/tone notes): \n-'

    return text_start + '\n-'.join(sentences)