import pandas as pd
import gc
import huggingface_hub
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
)
import torch
from .custom_trainer import CustomTrainer
from .training_utils import compute_metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import os 
import json

class NenClassifier():
    def __init__(
        self, model_path, data_path=None, text_column_name='Description', label_column_name='Types',
        model_name='distilbert/distilbert-base-uncased', test_size=0.2, num_labels=6, huggingface_token=None):

        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.test_size = test_size
        self.num_labels = num_labels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.label_dict = {0: 'Conjuration', 1: 'Emission', 2: 'Enhancement', 3: 'Manipulation', 4: 'Specialization', 5: 'Transmutation', 6: 'Unknown'}

        self.huggingface_token = huggingface_token
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        self.tokenizer = self.load_tokenizer()

        if not huggingface_hub.repo_exists(self.model_path):
            if data_path is None:
                raise ValueError('Data path is required to train the model')

            train_data, test_data, df = self.load_data(self.data_path)
            self.num_labels = len(self.label_dict)
            self.train_model(train_data, test_data)
    
        self.model = self.load_model(self.model_path)

    def load_model(self, model_path):
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def custom_pipeline(texts):
            if isinstance(texts, str):
                texts = [texts]
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).cpu().numpy()

            results = []
            for prob in probs:
                results.append([
                    {"label": self.label_dict[i], "score": float(p)}
                    for i, p in enumerate(prob)
                ])
            return results

        return custom_pipeline

    def train_model(self, train_data, test_data):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                    num_labels=self.num_labels,
                                                                    problem_type="multi_label_classification")

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(output_dir=self.model_path, learning_rate=2e-4,
                                          per_device_train_batch_size=8, per_device_eval_batch_size=8,
                                          num_train_epochs=5, weight_decay=0.01,
                                          eval_strategy='epoch', logging_strategy='epoch',
                                          push_to_hub=True)

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.set_device(self.device)
        trainer.train()

        del trainer, model
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def preprocess_function(self, tokenizer, examples):
        return tokenizer(examples['text_cleaned'], truncation=True)

    def load_data(self, data_path):
        df = pd.read_json(data_path, lines=True)
        df['text_cleaned'] = df['Name'].str.lower() + '. ' + df['Description'].str.lower()
        df = df.dropna()

        df[self.label_column_name] = df[self.label_column_name].apply(lambda x: x if isinstance(x, list) else [x])
        mlb = MultiLabelBinarizer()
        multi_hot = mlb.fit_transform(df[self.label_column_name])
        #self.label_dict = {i: cls for i, cls in enumerate(mlb.classes_)}

        df['label'] = multi_hot.tolist()

        try:
            df_train, df_test = train_test_split(df, test_size=self.test_size, stratify=multi_hot, random_state=42)
        except ValueError:
            print("Stratified split failed due to rare label combinations. Falling back to random split.")
            df_train, df_test = train_test_split(df, test_size=self.test_size, random_state=42)

        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        tokenized_train = train_dataset.map(lambda x: self.preprocess_function(self.tokenizer, x), batched=True)
        tokenized_test = test_dataset.map(lambda x: self.preprocess_function(self.tokenizer, x), batched=True)
        
        tokenized_train = tokenized_train.map(lambda x: {'labels': [float(i) for i in x['label']]})
        tokenized_test = tokenized_test.map(lambda x: {'labels': [float(i) for i in x['label']]})
        
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)
        tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)

        return tokenized_train, tokenized_test, df

    def load_tokenizer(self):
        if huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def postprocess(self, model_output, threshold=0.5):
        results = []
        for sample in model_output:
            labels = []
            for i, score_dict in enumerate(sample):
                if score_dict['score'] >= threshold:
                    labels.append(score_dict['label'])
            results.append(labels)
            
        return results

    def classify_nen(self, text):
        model_output = self.model(text)
        predictions = self.postprocess(model_output)
        updated_predictions = self.convert_list_to_string(predictions)
        
        return updated_predictions
    
    @staticmethod
    def convert_list_to_string(given_list):
        final_str = ''
        for item in given_list:
            list_str = ', '.join(item)
            final_str += list_str
            
        return final_str
