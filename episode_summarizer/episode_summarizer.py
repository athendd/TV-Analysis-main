import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import load_hunterxhunter_single_episode
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import os

class EpisodeSummarizer:
    
    def __init__(self):
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=600,          
            temperature=0.7,
            top_p=0.9,
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

    def summarize_chunk(self, chunk):
        """
        template = (
            
            Summarize this full Hunter x Hunter episode script into a paragraph long summary. Have the summary focus on major plot points with smooth transitions between each.
            This summary should give a good understanding of what happened in the episode from start to finish. Ensure that summary is complete and doesn't exceed the max token output.
            {text}
            
        )
        """
        
        template = (
        """
        Below is the full script from an episode of Hunter x Hunter. Write a single, well-structured paragraph that summarizes the entire episode.
        Focus on the key plot developments, character actions, and emotional turning points. Make sure the summary flows smoothly from beginning to end and captures the episode’s core story.
        Do not include unnecessary details, quotes, or dialogue formatting. Keep it concise and readable.
        {text}
        """
)


        prompt = PromptTemplate(template=template, input_variables=["text"])
        chain = prompt | self.llm
        
        return chain.invoke({"text": chunk})

    def get_episode_summary(self, data_path, save_path=None):
        if save_path and os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as file:
                return file.read()

        script = load_hunterxhunter_single_episode(data_path)            

        final_summary = self.summarize_chunk(script)
        final_summary = final_summary.strip()
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as file:
                file.write(final_summary)
        
        return final_summary
