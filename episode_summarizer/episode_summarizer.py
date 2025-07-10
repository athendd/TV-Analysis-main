import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from utils import load_hunterxhunter_single_episode

class EpisodeSummarizer():
    
    def __init__(self, similarity_threshold = 0.6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum").to(self.device)
        self.similarity_threshold = similarity_threshold
    
    
    def load_cleaned_dialogue(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        # Filter out filler lines
        return [line for line in lines if not line.lower().startswith(("next time", "click here", "back to"))]

    # Summarize a chunk of dialogue
    def summarize_chunk(self, dialogue_lines):
        text = " ".join(dialogue_lines)
        input_text = f"summarize: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # ✅ Move inputs to GPU or CPU


        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


            
    def get_episode_summary(self, data_path, save_path = None):
        """
        summary = ''

        if os.path.exists(save_path):
            with open(save_path, 'r', encoding = 'utf-8') as file:
                summary = file.read()

            return summary
        
        script = load_hunterxhunter_single_episode(data_path)
        
        print(type(script))

        summary = self.summarize_episode(script)
        
        
        if summary != '':
            if save_path is not None:
                with open(save_path, 'w') as file:
                    file.write(summary)
        else:
            summary = 'Unable to extract summarization from text'
        """
        lines = self.load_cleaned_dialogue(data_path)
        
        chunk_size = 20
        summaries = []

        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i + chunk_size]
            if chunk:
                summary = self.summarize_chunk(chunk)
                summaries.append(summary)

        # Combine all chunk summaries
        summary = " ".join(summaries)
            
        return summary
