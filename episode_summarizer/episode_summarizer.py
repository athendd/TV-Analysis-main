import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import load_hunterxhunter_single_episode
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import os

class EpisodeSummarizer:
    
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
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
        prompt = PromptTemplate(template=self.prompt_template, input_variables=["text"])
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
