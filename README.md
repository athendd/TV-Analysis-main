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
- Example: *Chimera Ant Arc* → dominant themes of “sacrifice” and “despair.”  

---

### 2. Character Network (NER + Graphs)  
- Extracts **character mentions** from subtitles with NER.  
- Builds **co-occurrence graphs**:  
  - Nodes = characters  
  - Edges = frequency of appearing together  
- Supports **arc-level** and **full-series** views.  
- Outputs **interactive HTML network graphs**.  
- Example: Kurapika’s network shifts from Leorio/Gon to Phantom Troupe in *Yorknew City Arc*.  

---

### 3. Nen Classifier (Custom Multi-Label Model)  
- Custom **MultiLabelClassifier** that assigns one or more Nen types:  
  - Enhancement, Conjuration, Manipulation, Emission, Transmutation, Specialization.  
- Supports **hybrid abilities** (e.g., Kurapika’s *Holy Chain* → Enhancement + Conjuration).  
- Input: free-text ability description.  
- Output: probability distribution → predicted Nen types.  
- Example: Gon’s *Scissors* → correctly classified as **Transmutation**.  

---

### 4. Episode Summarizer (LLM-Powered)  
- Built with **Mistral-7B-Instruct** for abstractive summarization.  
- Takes subtitle text and produces:  
  1. **Concise Summary**: one polished paragraph.  
  2. **Bullet List of Events**: chronological step-by-step list.  
- Example: Episode 116 → bullet list highlights Killua saving Gon, Gon confronting Neferpitou, Komugi’s state.  

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
- Example: Asking Gon *“What’s your dream?”* →  
  *“Hey there! I’ve always wanted to become a Hunter, just like my dad, Ging. It’s all about exploring the world, facing challenges, and growing stronger. That’s my dream, to keep on pushin’ forward!”*  

---

## 🛠️ Installation  

Clone the repo:  
```bash
git clone https://github.com/yourusername/hxh-ai-chatbot.git
cd hxh-ai-chatbot
