from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:

    def __init__(self, model_name = 'BAAI/bge-small-en-v1.5'):
        self.embedder = HuggingFaceEmbeddings(model_name = model_name, model_kwargs = {'device': 'cpu'}, encode_kwargs = {'normalize_embeddings': True})
        
    def embed_documents(self, documents):
        return self.embedder.embed_documents(documents)
    
    def embed_query(self, query, instructions = "Represent this question for retrieving relevant passages about Hunter x Hunter characters, their abilities, equipment, and plot details."):
        return self.embedder.embed_query(f'{instructions} {query}')
