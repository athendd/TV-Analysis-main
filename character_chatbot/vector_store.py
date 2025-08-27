import os
import json
from typing import List, Dict, Optional
from pathlib import Path
from .util_methods import load_dict

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever

from .embedder import Embedder
from .character_data_retriever import CharacterRetriever, SECTION_PROFILE


"""
Wraps Embedder class to match LangChain's embeddings interace so 
FAISS can get LangChain comptatible embeddings 
"""
class _LangChainEmbeddingsAdapter:

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    """
    Embedds the given query 

    Args:
        -text (str): Given piece of text to embed
    Returns:
        -(list): A list of floats that represent the embedding
    """
    def embed_query(self, text):
        return self._embedder.embed_query(text)

    """
    Embedds the given documents

    Args:
        -text (str): Given piece of text to embed
    Returns:
        -(list): A list of floats that represent the embedding
    """
    def embed_documents(self, texts):
        return self._embedder.embed_documents(texts)

"""
Represents a vector store containing data on characters from HunterxHunter.
Capable of crating the vector store and retrieving data from the vector store
"""
class VectorStore:

    def __init__(self, embedder: Embedder, documents = None, vector_store_path = None, profile_map = None):
        if not vector_store_path:
            raise ValueError('vector_store_path must be provided')

        self.embedder = embedder
        self._lc_embeddings = embedder.embedder 
        self.documents = documents 
        self.vector_store_path = str(Path(vector_store_path).resolve())
        self.vector_store = None
        self.bm25_retriever = None
        self.profile_map = profile_map 

        """
        Checks to see if the given vector store path contians all the necessary files

        Args:
            -dirpath (str): The given directory path that is supposed to contain the vector store
        Returns:
            -(bool): Returns true if the directory path has the vector store, else returns false
        """
        def _has_faiss_files(dirpath):
            path = Path(dirpath)
            has_index = (path / 'index.faiss').exists() 
            has_meta  = (path / 'index.pkl').exists() 
            has_profile = (path / 'profile_map').exists()

            return path.is_dir() and has_index and has_meta and has_profile

        if _has_faiss_files(self.vector_store_path):
            self.load_vector_store()

        else:
            if not self.documents:
                raise ValueError(
                    "No complete FAISS artifacts found at "
                    f"'{self.vector_store_path}'.\n"
                    "Expected at least 'index.faiss' and 'index.pkl'.\n"
                    "→ Point to the correct folder OR provide `documents` so I can build and save the store."
                )
            self.build_vector_store()

    """
    Builds the profile map and index data that represents the vector store and saves them to the given directory
    """
    def build_vector_store(self):
        self.vector_store = FAISS.from_documents(self.documents, self._lc_embeddings)
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)

        self.save_vector_store_to_disk()
        self.save_profile_map()

    """
    Saves the vector store data as pkl and faiss files to the given directory
    """
    def save_vector_store_to_disk(self):
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)

    def load_vector_store(self):
        self.vector_store = FAISS.load_local(
            self.vector_store_path,
            self._lc_embeddings,
            allow_dangerous_deserialization=True
        )
        docs = []
        try:
            self.profile_map = load_dict(os.path.join(self.vector_store_path, 'profile_map'))
            if not self.profile_map:
                raise Exception
            
            ds = getattr(self.vector_store, 'docstore', None)
            if ds is not None:
                # LangChain's InMemoryDocstore keeps items in a _dict
                if hasattr(ds, '_dict') and isinstance(ds._dict, dict):
                    docs = [v for v in ds._dict.values() if isinstance(v, Document)]
                else:
                    # Fallback path: use index_to_docstore_id mapping
                    ids = []
                    if hasattr(self.vector_store, 'index_to_docstore_id'):
                        ids = list(self.vector_store.index_to_docstore_id.values())
                    if ids:
                        fetched = ds.mget(ids)
                        docs = [d for d in fetched if isinstance(d, Document)]
        except Exception as e:
            print(f'[VectorStore] Warning: could not rebuild docs from FAISS docstore: {e}')

        # If we recovered any docs, keep them for BM25; otherwise disable BM25
        if docs:
            self.documents = docs
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        else:
            print('[VectorStore] No documents available; disabling BM25 retriever.')
            self.bm25_retriever = None

    def perform_search(self, query: str, character_name: Optional[str], k: int = 10) -> List[Document]:
        if not query or not isinstance(query, str):
            raise ValueError('Query must be a non-empty string.')
        if self.vector_store is None or self.bm25_retriever is None:
            raise RuntimeError('Vector store not initialized.')

        # FAISS with metadata filter (tight)
        faiss_retriever = self.vector_store.as_retriever(
            search_kwargs={'k': k},
            filter={'name': character_name} if character_name else None,
        )

        # BM25 (broad) — increase recall
        self.bm25_retriever.k = max(k * 3, 10)

        # Fuse BM25 + FAISS
        ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6],
            search_kwargs={'k': max(k * 3, 10)},  
        )

        docs = ensemble.get_relevant_documents(query)

        # Post-filter by character name, but backfill to keep k
        if character_name:
            filtered = [d for d in docs if d.metadata.get("name") == character_name]
            if len(filtered) < k:
                need = k - len(filtered)
                spill = [d for d in docs if d not in filtered][:need]
                filtered.extend(spill)
            return filtered[:k]

        return docs[:k]

    # ---------- Profiles ----------

    def get_profile(self, character_name: str) -> Dict[str, str]:
        """
        O(1) lookup using profile_map; falls back to a tiny scoped FAISS search.
        """
        if not character_name:
            return {}
        if character_name in self.profile_map:
            return self.profile_map[character_name]

        if self.vector_store is None:
            return {}
        hits = self.vector_store.similarity_search(
            query="profile persona voice",
            k=1,
            filter={"name": character_name, "section": SECTION_PROFILE},
        )
        if hits:
            # If desired, parse hits[0].page_content to split persona vs voice.
            return {"persona_card": hits[0].page_content, "voice_cues": ""}
        return {}
    
    def save_profile_map(self):
        os.makedirs(self.vector_store_path, exist_ok=True)
        full_file_path = os.path.join(self.vector_store_path, 'profile_map')
        try:
            with open(full_file_path, 'w') as f:
                json.dump(self.profile_map, f, indent=4)
            print(f"Dictionary successfully saved to {full_file_path}")
        except IOError as e:
            print(f"Error saving file: {e}")

if __name__ == '__main__':
    # Build docs from your character JSONs
    character_retriever = CharacterRetriever(r'Data\Character_Data_for_Chatbot')
    documents = character_retriever.create_character_documents()
    profile_map = character_retriever.build_profile_map()

    # Your Embedder (from your snippet)
    embedder = Embedder()

    # Create or load the store
    vector_store = VectorStore(
        embedder=embedder,
        documents=documents,  # also used to build BM25
        vector_store_path=r'Data\vector_stores\vector_store_one',
        profile_map=profile_map,
    )

    # quick checks
    print("Profile (example):", vector_store.get_profile("Gon"))
    results = vector_store.perform_search("Jajanken and Enhancement", "Gon", k=5)
    for r in results:
        print(r.metadata, "=>", r.page_content[:120], "…")
