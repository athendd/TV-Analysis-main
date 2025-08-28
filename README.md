# Hunter x Hunter AI Analysis & Character Chatbot  

An **AI-powered toolkit** for exploring the world of *Hunter x Hunter* through NLP and LLMs.  
This project combines **zero-shot classifiers, custom multi-label models, knowledge-grounded chatbots, and retrieval-augmented summarization** to analyze themes, characters, and episodes in ways that fans and researchers alike can use.  

---

## Features  

### 1. Theme Classifier (Zero-Shot)  
- Uses **zero-shot text classification** to label subtitles with themes (e.g., *friendship*, *betrayal*, *sacrifice*).  
- Works at two levels:  
  - **Entire Series**: classify themes across all subtitles.  
  - **Arc-Specific**: classify themes in arcs like *Chimera Ant* or *Yorknew City*.  
- Outputs aggregated counts visualized in bar plots for fast comparison.  
- Example: *Greed Island Arc* →   
<img width="3752" height="982" alt="Greed Island Themes" src="https://github.com/user-attachments/assets/357c9fc8-fb9b-4207-83ae-8bdd564cd3de" />

---

### 2. Character Network (NER + Graphs)  
- Extracts **character mentions** from subtitles with NER.  
- Builds **co-occurrence graphs**:  
  - Nodes = characters  
  - Edges = frequency of appearing together  
- Supports **arc-level** and **full-series** views.  
- Outputs **interactive HTML network graphs**.  
- Example: *Zoldyck Family Arc*.  
<img width="3680" height="1610" alt="Zoldyck Family Arc Network" src="https://github.com/user-attachments/assets/89687a84-63e6-4765-82cf-0568d8592312" />

---

### 3. Nen Classifier (Custom Multi-Label Model)  
- Custom **MultiLabelClassifier** that assigns one or more Nen types:  
  - Enhancement, Conjuration, Manipulation, Emission, Transmutation, Specialization.  
- Supports **hybrid abilities** (e.g., Kurapika’s *Holy Chain* → Enhancement + Conjuration).
- Performed **word level text augmentation** on nen abilities through **synonym replacement** to create enough data to train the classifier
- Input: free-text ability description.  
- Output: probability distribution → predicted Nen types.  
- Example: Gon’s *Scissors* →   
<img width="3702" height="1137" alt="Scissors Classification" src="https://github.com/user-attachments/assets/59446150-0a4d-4062-8720-d4a11d793450" />

---

### 4. Episode Summarizer (LLM-Powered)  
- Built with **Mistral-7B-Instruct** for abstractive summarization.  
- Takes subtitle text and produces:  
  1. **Concise Summary**: one polished paragraph.  
  2. **Bullet List of Events**: chronological step-by-step list.  
- Example: Episode 74 →
<img width="3700" height="1085" alt="Concise Episode Summary" src="https://github.com/user-attachments/assets/26c2791c-b85b-48e1-acd1-67140e0b00d6" />

---

### 5. Character Chatbot (RAG + Persona Conditioning)  
- Interactive **character chatbots** for **Gon Freecss, Killua Zoldyck, and Kurapika**.  
- **Data Sources**:  
  - Wiki-scraped: background, abilities, affiliations, equipment, arc plots.  
  - Quotes analyzed for **emotion + tone**, forming personality profiles.  
- **Hybrid Retrieval**:  
  - **FAISS embeddings** + **BM25 keyword search** combined with an **Ensemble Retriever**.  
  - Ensures canon-accurate responses.  
- **Persona Conditioning**:  
  - Persona card: character traits, motivations.  
  - Voice cues: cadence, vocabulary, tone, formality.  
  - Responses always sound “in-character.”  
- Example: Asking Gon *“Who is your closest friend?”* →  
<img width="1535" height="1307" alt="Gon's Best Friend" src="https://github.com/user-attachments/assets/f67f4131-7e6f-435c-86ce-8e12f09ce874" />

---

## Installation

Clone the repo:
```bash
git clone https://github.com/yourusername/TV-Analysis-main.git
```

## Steps to Run Program

1. Install the requirements
```bash
pip install -r requirements.txt
```
2. Setup Huggingface token as environment variable for access to Huggingface
```bash
touch .env
echo "huggingface_token=your_hf_token_here" >> .env
```
3. Run gradio.py

---

## Tech Stack

**LLMs**: Mistral-7B-Instruct
**Embeddings**: BAAI/bge-small-en-1.5
**Vector Store**: FAISS (dense retrieval) + BM25 (sparse retrieval)
**Interface**: Gradio
**NER**: spaCy + custom pipelines
**Classification**: Huggingface Transformers
