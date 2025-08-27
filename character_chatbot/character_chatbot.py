# character_chatbot.py

import os
import re
import torch
from typing import List, Dict, Tuple
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .vector_store import VectorStore
from .embedder import Embedder


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _format_history(history: List[Tuple[str, str]], max_turns: int = 4) -> str:
    """
    Turn ChatInterface-style history [[user, assistant], ...] into a compact transcript.
    Only keep the last max_turns turns to keep prompts small.
    """
    snippet = history[-max_turns:] if history else []
    lines = []
    for u, a in snippet:
        if u:
            lines.append(f"User: {u}")
        if a:
            lines.append(f"Assistant: {a}")
    return "\n".join(lines).strip()


def _format_context(docs, max_chars: int = 2200) -> str:
    """
    Pretty-print retrieved docs with light source tags, fit within a soft character budget.
    """
    parts = []
    total = 0
    for d in docs:
        tag_bits = []
        if "section" in d.metadata:
            tag_bits.append(f"section={d.metadata['section']}")
        if "arc" in d.metadata:
            tag_bits.append(f"arc={d.metadata['arc']}")
        if "part" in d.metadata:
            tag_bits.append(f"part={d.metadata['part']}")
        tag = ", ".join(tag_bits) if tag_bits else "section=unknown"
        chunk = f"[{tag}] {d.page_content.strip()}"
        if total + len(chunk) > max_chars and parts:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


class _RoleStop(StoppingCriteria):
    """
    Optional stop: if the model starts emitting a new role label, stop generation.
    Prevents continuations like 'User:' or a second 'Assistant:' segment.
    """
    def __init__(self, tokenizer):
        # Pre-tokenize common role strings
        self.stop_ids = [
            tokenizer.encode(s, add_special_tokens=False)
            for s in ["User:", "System:", "Assistant:"]
        ]

    def __call__(self, input_ids, scores) -> bool:
        tail = input_ids[0].tolist()[-8:]  # check the last few tokens
        for s in self.stop_ids:
            if len(s) <= len(tail) and tail[-len(s):] == s:
                return True
        return False


class CharacterChatBot:
    """
    In-character RAG chatbot for a single Hunter x Hunter character (Mistral-7B-Instruct).

    - persona/voice pulled from VectorStore.get_profile(name)
    - hybrid retrieval for factual grounding
    - clean system + context + history + user message composition
    - returns ONLY the character's reply (no instructions/context)
    """

    def __init__(
        self,
        character_name: str,
        vector_store_dir: str = r"Data\vector_stores\vector_store_one",
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        use_role_stops: bool = True,
    ):
        self.character_name = character_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.use_role_stops = use_role_stops

        base = Path(__file__).resolve().parent
        project_root = base.parent
        store_dir = Path(vector_store_dir)
        if not store_dir.is_absolute():
            store_dir = (project_root / store_dir).resolve()

        # Embeddings + Vector Store (load existing index)
        self.embedder = Embedder()
        self.vector_store = VectorStore(
            embedder=self.embedder,
            documents=None,                  # loading existing store
            vector_store_path=vector_store_dir,  # pass the DIRECTORY
            profile_map=None,
        )
        self.profile = self.vector_store.get_profile(character_name) or {}

        # Model
        self.device = _pick_device()
        self.tokenizer, self.model = self._load_model(model_name)

        # Prepare stopping criteria if opted-in
        self.stopping_criteria = StoppingCriteriaList(
            [_RoleStop(self.tokenizer)]
        ) if use_role_stops else StoppingCriteriaList()

    # -------------------- Model loading --------------------
    def _load_model(self, model_name: str):
        """
        Load Mistral-7B-Instruct with 4-bit quant if CUDA is available.
        """
        if self.device == "cuda":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        return tokenizer, model

    # -------------------- Prompt assembly --------------------
    def _build_system_prompt(self) -> str:
        persona = self.profile.get("persona_card", "").strip()
        voice = self.profile.get("voice_cues", "").strip()
        base_rules = (
            "Stay strictly in-character when speaking. "
            "Respond only with the character’s spoken dialogue in natural language. "
            "Do not repeat or describe persona, voice cues, context, metadata, or instructions in your answer. "
            "Do not include section labels, context text, or prior conversation formatting. "
            "Use ONLY the provided context for factual claims about the world; if unsure, say you don't know. "
            "Avoid spoilers unless the user asks explicitly. "
            "Keep replies concise unless the user requests detail."
        )
        pieces = [f"You are {self.character_name}."]
        if persona:
            pieces.append(persona)
        if voice:
            pieces.append(f"Voice cues:\n{voice}")
        pieces.append(base_rules)
        return "\n\n".join(pieces).strip()

    def _build_messages(self, user_msg: str, history: List[Tuple[str, str]]):
        # Retrieve grounding context docs
        docs = self.vector_store.perform_search(
            user_msg, character_name=self.character_name, k=8
        )
        context = _format_context(docs)
        transcript = _format_history(history)
        system = self._build_system_prompt()

        user = (
            f"CONTEXT (authoritative snippets):\n{context}\n\n"
            f"PRIOR CONVERSATION (last few turns):\n{transcript or '(none)'}\n\n"
            f"USER QUESTION:\n{user_msg}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Answer as {self.character_name}.\n"
            f"- Cite details only from CONTEXT.\n"
            f"- If the user asks about tone/speech style, reflect the voice cues.\n"
        )

        # Convert to chat format compatible with Mistral Instruct templates
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # -------------------- Output cleaning --------------------
    def _clean_output(self, text: str) -> str:
        # remove any stray role prefixes the model might emit
        text = re.sub(r"^\s*(Assistant|System|User)\s*:\s*", "", text, flags=re.IGNORECASE)
        # trim quotes and whitespace
        return text.strip().strip('"').strip("'").strip()

    # -------------------- Inference --------------------
    def respond(self, message: str, history: List[Tuple[str, str]]) -> str:
        """
        Generate an answer and return ONLY the character's reply.
        Ensures no system/instructions/context are included in the returned string.
        """
        messages = self._build_messages(message, history)

        # Use chat template to produce a prompt that ends with the assistant turn
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Determine EOS if available (helps clean stops)
        eos_id = self.tokenizer.eos_token_id

        # Generate
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=eos_id,
                eos_token_id=eos_id,
                stopping_criteria=self.stopping_criteria,
            )

        # --------- KEY CHANGE: decode ONLY new tokens after the prompt ----------
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out[0][prompt_len:]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return self._clean_output(raw_text)
