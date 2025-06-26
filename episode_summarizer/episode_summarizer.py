from transformers import BartForConditionalGeneration, BartTokenizer
import torch

#Figure out what num beams means and length penalty means
#See about using Sentence Embedding Clustering for better overlapping of scenes and splitting

def load_hunterxhunter_single_episode(dataset_path):
    episode_num = int(dataset_path.split("\\")[-1].split(" ")[-1].split(".")[0])
    lines = []
    
    with open(dataset_path, 'r', encoding = 'utf-8') as file:
        lines = file.readlines()
        
        if episode_num >= 1 and episode_num < 22:
            lines = lines[8:]
    
    return lines

script = load_hunterxhunter_single_episode(r'Data\HunterxHunterSubtitles\Hunter Exam Arc\Episode 1.txt')

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Remove all /n from the script
script = [s.replace('\n', '') for s in script]

sentence_chunks = []
curr_chunk = ''
len_tokens = 0
for i in range(len(script)):
    len_tokens += len(tokenizer.tokenize(script[i] + curr_chunk))
    if len_tokens >= 1024:
        sentence_chunks.append(curr_chunk.strip())
        curr_chunk = script[i-2] + '. ' + script[i-1] + '. ' + script[i] + '. '
    else:
        curr_chunk += script[i].strip() + '. '

chunk_summaries = []

for sentence_chunk in sentence_chunks:
    inputs = tokenizer.encode(sentence_chunk, return_tensors = 'pt', truncation = True, max_length = 1024).to(device)
    summary_ids = model.generate(
            inputs,
            max_length= 150,
            min_length= 50,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    chunk_summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens = True))

chunks_combined = ' '.join(chunk_summaries)

inputs = tokenizer.encode(sentence_chunk, return_tensors = 'pt', truncation = True, max_length = 1024).to(device)
summary_ids = model.generate(
        inputs,
        max_length= 200,
        min_length= 80,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
print(final_summary)
    
