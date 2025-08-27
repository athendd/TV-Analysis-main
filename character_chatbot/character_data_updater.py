import re
import os
from dataclasses import dataclass
from typing import List, Dict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, pipeline
from seralizer import JsonSerializer
from util_methods import load_dict, convert_quotes_to_sentences

SEQ2SEQ_MAX_INPUT_TOKENS = 1024           
LONG_MAX_INPUT_TOKENS = 16384             

@dataclass
class SummTask:
    key: str
    instructions: str
    text: str
    max_new_tokens: int


class CharacterUpdater:

    descs_dict = {
        'personality_prompt': (
            "Personality (≤128 words; 1–2 sentences; neutral; no lists):\n\n"
            "{SOURCE_TEXT}\n\n"
            "Summary:"
        ),

        'speech_prompt': (
            "You are analyzing how a fictional character speaks.\n\n"
            "TASK:\n"
            "Given QUOTES, Emotion, Tone, and Personality, describe the character’s *speech style*. "
            "Focus ONLY on delivery and manner of speaking (cadence, pacing, vocabulary, formality, punctuation habits, emotional register).\n"
            "- Do NOT copy or paraphrase the quotes.\n"
            "- Do NOT include plot or story content.\n"
            "- Return 4–6 bullet points.\n"
            "- Each bullet MUST start with '- ' and be ≤15 words.\n"
            "- Cover at least cadence, vocabulary, tone, and formality.\n\n"
            "EXAMPLE:\n"
            "QUOTES: \"Heh. What a fun toy... I'll save you for later.\"\n"
            "OUTPUT:\n"
            "- Cadence: abrupt bursts broken by long pauses.\n"
            "- Vocabulary: playful menace, childish metaphors for violence.\n"
            "- Punctuation: ellipses signal teasing hesitation.\n"
            "- Tone: flirtatious taunting hides hostility.\n"
            "- Formality: low; casual slangy interjections.\n\n"
            "SOURCE:\n{SOURCE_TEXT}\n\n"
            "OUTPUT:"
        ),

        'appearance_prompt': (
            "Appearance paragraph (≤120 words). Start with stable traits, then arc variations. No analysis or meta.\n\n"
            "{SOURCE_TEXT}\n\n"
            "Description:"
        ),

        'plot_prompt': (
            "Event-based arc recap: 8–14 short sentences, chronological; each sentence is an action/decision/consequence "
            "the character experiences; include climax and final outcome.\n\n"
            "{SOURCE_TEXT}\n\n"
            "Recap:"
        ),

        'final_plot_merge_prompt': (
            "Merge these notes into one chronological event recap focused on the character. "
            "Keep event-per-sentence style where possible, add brief transitions (Then, Later, Afterwards, Meanwhile), "
            "and ensure the climax and final consequences appear.\n\n"
            "{SOURCE_TEXT}\n\n"
            "Final recap:"
        ),

        'nen_overview_prompt': (
            "Nen overview paragraph (≤160 words): primary type, secondary proficiencies, staple techniques, strategic trademarks. "
            "Neutral, no lists, only terms present in source.\n\n"
            "{SOURCE_TEXT}\n\n"
            "Overview:"
        ),

        'abilities_overview_prompt': (
            "Abilities overview (≤160 words): general strengths, combat style, reputation, broad Nen proficiency; "
            "what makes them dangerous/unique. No detailed mechanics. Paragraph only.\n\n"
            "{SOURCE_TEXT}\n\n"
            "Summary:"
        ),

        'personality_traits_prompt': (
            "List 6–12 enduring personality traits (one word each). No physical traits or moods.\n\n"
            "{SOURCE_TEXT}\n\n"
            "Traits:"
        ),

        'background_prompt': (
            "Background paragraph (≤200 words), objective, chronological.\n\n"
            "{SOURCE_TEXT}\n\n"
            "Background:"
        )
    }

    def __init__(self, file_path, short_model_name = 'facebook/bart-large-cnn', voice_cues_model_name = 'mistralai/Mistral-7B-Instruct-v0.3'):
        self.file_path = file_path
        self.serializer = JsonSerializer(os.path.dirname(self.file_path))
        self.voice_cues_model_name = voice_cues_model_name
        self.short_model_name = short_model_name

        self._load_short_model()
        self._load_voice_cues_model()

        self._instr_keywords = [
            "summarize", "summary:", "recap:", "write", "rule", "rules", "source", "output",
            "instruction", "format", "chunk", "merge", "final", "requirements", "ensure",
            "include", "begin", "start", "now:", "use only", "no meta", "headings", "list",
            "bullets", "each sentence", "cover the entire", "aim for", "sentences", "paragraph",
            "the only difference", "the summary will", "write the summary", "write 8", "8–14", "8-14",
            "chronological order", "cover the whole", "add brief transitions"
        ]

    def _load_short_model(self): 
        use_cuda = torch.cuda.is_available() 
        self.tokenizer = AutoTokenizer.from_pretrained(self.short_model_name, use_fast=True) 
        self.short_model = AutoModelForSeq2SeqLM.from_pretrained(self.short_model_name, torch_dtype=torch.float16 if use_cuda else torch.float32, low_cpu_mem_usage=use_cuda ) 
        if use_cuda: 
            self.short_model.to("cuda") 

        self.model_max_positions = getattr(self.short_model.config, "max_position_embeddings", 1024) 
        self.tokenizer.model_max_length = self.model_max_positions 
        self.pipe = pipeline( 'summarization', model=self.short_model, tokenizer=self.tokenizer, device=0 if use_cuda else -1 ) 
        self.bad_words_short = self._make_bad_words(self.tokenizer)

    def _load_voice_cues_model(self):
        use_cuda = torch.cuda.is_available()
        self.voice_cues_tokenizer = AutoTokenizer.from_pretrained(self.voice_cues_model_name, use_fast=True)
        self.voice_cues_model = AutoModelForCausalLM.from_pretrained(
            self.voice_cues_model_name,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            device_map="auto" if use_cuda else None
        )

        self.voice_cues_pipe = pipeline(
            task="text-generation",
            model=self.voice_cues_model,
            tokenizer=self.voice_cues_tokenizer,
            device=0 if use_cuda else -1
        )

    def create_updated_character_data(self):
        character_data = load_dict(self.file_path)
        if not character_data:
            return

        updated_quotes = convert_quotes_to_sentences(character_data.get('Quotes', []))
        personality_and_quotes = f"{updated_quotes} \n\n PERSONALITY: \n{character_data['Personality']}"

        tasks: List[SummTask] = [
            SummTask('__persona_body__', self.descs_dict['personality_prompt'], character_data.get('Personality', ''), 128),
            SummTask('appearance', self.descs_dict['appearance_prompt'], character_data.get('Appearance', ''), 128),
            SummTask('background', self.descs_dict['background_prompt'], character_data.get('Background', ''), 200)
        ]

        nen_desc = character_data.get('Abilities & Powers', {}).get('Nen', {}).get('Description', '')
        ability_desc = character_data.get('Abilities & Powers', {}).get('Description', '')
        tasks.append(SummTask('nen_summary', self.descs_dict['nen_overview_prompt'], nen_desc, 160))
        tasks.append(SummTask('abilities_summary', self.descs_dict['abilities_overview_prompt'], ability_desc, 160))

        results = self._summarize_batch_short(tasks)
        voice_cues_result = self._summarize_voice_cues(personality_and_quotes)

        updated_character_data = {}
        updated_character_data['name'] = character_data.get('Name', 'Unknown')
        updated_character_data['affiliations'] = {
            'current': self._split_csv(character_data.get('Affiliation', '')),
            'previous': self._split_csv(character_data.get('Previous affiliation', ''))
        }
        updated_character_data['status'] = character_data.get('Status', '')
        updated_character_data['quotes'] = character_data.get('Quotes', [])
        updated_character_data['relationships'] = character_data.get('Relatives', {})
        updated_character_data['equipment'] = character_data.get('Equipment', {})
        updated_character_data['birthday'] = character_data.get('Birthday', '')
        updated_character_data['nen_type'] = character_data.get('Type', '')
        updated_character_data['plot'] = character_data.get('Plot', {})

        updated_character_data['non-nen_abilities'] = self.get_character_abilites(character_data.get('Abilities & Powers', {}))
        updated_character_data['nen_abilities'] = self.get_character_abilites(character_data.get('Abilities & Powers', {}).get('Nen', {}))

        updated_character_data['persona_card'] = (
            f"You are {updated_character_data['name']}, a {updated_character_data['nen_type']} Nen user from the anime Hunter x Hunter. "
            f"{results.get('__persona_body__', '')}"
        )
        updated_character_data['background'] = results.get('background', '')
        updated_character_data['appearance'] = results.get('appearance', '')
        updated_character_data['nen_summary'] = results.get('nen_summary', '')
        updated_character_data['abilities_summary'] = results.get('abilities_summary', '')
        updated_character_data['voice_cues'] = voice_cues_result

        self.serializer.save(updated_character_data['name'] + '_Updated', updated_character_data)

    def _summarize_batch_short(self, tasks):
        output: Dict[str, str] = {}
        if not tasks:
            return output

        groups = self._bucket_by_tokens(tasks, bucket_sizes=(64, 96, 128, 160))

        for bucket_size, group in groups.items():
            inputs = []
            for t in group:
                prompt = t.instructions.replace("{SOURCE_TEXT}", t.text or "")
                prompt = self._trim_prompt_to_short_model(prompt)
                inputs.append(prompt)

            max_len = min(max(32, bucket_size), 160)
            min_len = min(10, max_len - 1)

            with torch.inference_mode():
                results = self.pipe(
                    inputs,
                    max_length=max_len,
                    min_length=min_len,
                    truncation=True,
                    num_beams=3,
                    no_repeat_ngram_size=3,
                    bad_words_ids=self.bad_words_short
                )

            for task, res in zip(group, results):
                text = (res.get('summary_text') or "").strip()
                text = self._sanitize_summary(text, min_sent=1, max_sent=8)
                output[task.key] = text

        return output

    def _summarize_voice_cues(self, voice_cues):
        voice_cues_prompt = self.descs_dict['speech_prompt'].replace("{SOURCE_TEXT}", voice_cues or '')

        out = self.voice_cues_pipe(
            voice_cues_prompt,
            max_new_tokens=160,
            do_sample=False,
            num_beams=5,
            repetition_penalty=1.05,
            eos_token_id=self.voice_cues_tokenizer.eos_token_id,
            pad_token_id=self.voice_cues_tokenizer.pad_token_id,
        )[0]["generated_text"]

        if out.startswith(voice_cues_prompt):
            out = out[len(voice_cues_prompt):].lstrip()

        return out

    def _sanitize_summary(self, text: str, min_sent: int = 8, max_sent: int = 14) -> str:
        if not text:
            return ""

        trash_patterns = [
            r'\bTASK\b', r'\bRULES\b', r'\bSOURCE\b', r'\bOUTPUT\b',
            r'INSTRUCTIONS?:', r'FORMAT:',
            r'</?[\w:/-]+>',    
            r'/TASK', r'/RULES', r'/SOURCE', r'/OUTPUT',
            r'Chunk summaries?:', r'\bFinal recap:\b', r'\bRecap:\b', r'\bSummary:\b', r'\bOverview:\b',
        ]
        clean = text
        for pat in trash_patterns:
            clean = re.sub(pat, '', clean, flags=re.IGNORECASE)

        clean = re.sub(r'\s+', ' ', clean).strip()

        parts = re.split(r'(?<=[\.\?\!])\s+', clean)
        parts = [p.strip() for p in parts if p.strip()]

        def looks_instructional(s):
            s_low = s.lower()
            return any(k in s_low for k in self._instr_keywords)

        parts = [p for p in parts if not looks_instructional(p)]

        if not parts:
            return ""

        refined = []
        for p in parts:
            refined.extend(re.split(r'\s*[\u2022\-–]\s*', p))
        parts = [p.strip(' .;:') for p in refined if p.strip(' .;:')]

        if len(parts) >= min_sent:
            parts = parts[:max_sent]

        parts = [p if p.endswith(('.', '!', '?')) else p + '.' for p in parts]

        return ' '.join(parts)

    def _make_bad_words(self, tok):
        bad_pieces = [
            "TASK", "task", "Rules", "RULES", "rules", "Source", "SOURCE", "source",
            "Output", "OUTPUT", "output",
            "<TASK>", "</TASK>", "<RULES>", "</RULES>", "<SOURCE>", "</SOURCE>",
            "<OUTPUT>", "</OUTPUT>",
            "INSTRUCTIONS:", "Instructions:", "instructions:", "FORMAT:", "Format:", "format:",
            "/TASK", "/RULES", "/SOURCE", "/OUTPUT",
            "Chunk summaries:", "chunk summaries:", "Final recap:", "Recap:", "Summary:", "Overview:",
            "Write the summary", "Write 8", "8–14 sentences", "8-14 sentences", "Only use the source",
            "No meta", "no meta", "Aim for", "Begin:", "Start now:", "Write the final"
        ]
        ids = []
        for w in bad_pieces:
            try:
                wi = tok.encode(w, add_special_tokens=False)
                if wi:
                    ids.append(wi)
            except Exception:
                pass

        return ids

    def _trim_prompt_to_short_model(self, prompt):
        reserve = 8
        limit = max(16, self.model_max_positions - reserve)
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=limit,
            add_special_tokens=True,
            return_attention_mask=False
        )

        return self.tokenizer.decode(enc["input_ids"], skip_special_tokens=True)

    def _trim_prompt_to_voice_cues_model(self, prompt):
        model_max = getattr(self.voice_cues_model.config, "n_positions", None) \
                or getattr(self.voice_cues_model.config, "max_position_embeddings", None) \
                or getattr(self.voice_cues_tokenizer, "model_max_length", 1024)
        cap = max(128, min(int(model_max) - 32, 2048))  

        enc = self.voice_cues_tokenizer(
            prompt,
            truncation=True,
            max_length=cap,
            add_special_tokens=True,
            return_attention_mask=False
        )

        return self.voice_cues_tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
    
    @staticmethod
    def _token_len(tokenizer, text):
        return len(tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])

    @staticmethod
    def _chunk_by_tokens(tokenizer, text, max_tokens, overlap_tokens = 64):
        enc = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        ids = enc["input_ids"]
        chunks = []
        i = 0
        while i < len(ids):
            j = min(i + max_tokens, len(ids))
            chunk_ids = ids[i:j]
            chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
            if j == len(ids):
                break
            i = max(0, j - overlap_tokens)

        return chunks

    @staticmethod
    def get_character_abilites(ability_dict):
        updated_ability_dict = {}
        for key in ability_dict.keys():
            if key not in ['Nen', 'Description']:
                updated_ability_dict[key] = ability_dict[key]
        return updated_ability_dict

    @staticmethod
    def _split_csv(s):
        return [x.strip() for x in s.split(',') if x.strip()]

    @staticmethod
    def _split_bullets(text):
        lines = (text or '').splitlines()
        output = []
        for ln in lines:
            t = ln.strip().lstrip('-*•').strip()
            if t:
                output.append(t)
        if len(output) <= 1 and ',' in (output[0] if output else ''):
            output = [x.strip() for x in output[0].split(',') if x.strip()]
        return output

    @staticmethod
    def _build_prompt(instr_template, source_text):
        return instr_template.replace("{SOURCE_TEXT}", source_text or "")

    @staticmethod
    def _bucket_by_tokens(tasks, bucket_sizes):
        buckets = {b: [] for b in bucket_sizes}
        for task in tasks:
            chosen = min((b for b in bucket_sizes if b >= task.max_new_tokens), default=max(bucket_sizes))
            buckets[chosen].append(task)
        return {k: v for k, v in buckets.items() if v}
    
    def _sanitize_bullets(self, text, min_items=3, max_items=6, source = ''):
        if not text:
            return ""

        trash_patterns = [
            r'\bTASK\b', r'\bRULES\b', r'\bSOURCE\b', r'\bOUTPUT\b',
            r'INSTRUCTIONS?:', r'FORMAT:',
            r'</?[\w:/-]+>', r'/TASK', r'/RULES', r'/SOURCE', r'/OUTPUT',
            r'\bFinal recap:\b', r'\bRecap:\b', r'\bSummary:\b', r'\bOverview:\b', r'\bBullets:\b'
        ]
        clean = text
        for pat in trash_patterns:
            clean = re.sub(pat, '', clean, flags=re.IGNORECASE)

        clean = re.sub(r'\r\n?', '\n', clean)
        lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
        bullets = []
        for ln in lines:
            if re.match(r'^[-•–]\s+', ln):
                ln = re.sub(r'^[•–]\s+', '- ', ln)
            elif not ln.startswith('- '):
                ln = f"- {ln}"

            low = ln.lower()
            if any(k in low for k in self._instr_keywords):
                continue

            words = ln[2:].split()
            if len(words) > 18:
                ln = "- " + " ".join(words[:18])

            if source and self._ngram_overlap(source, ln) > 0.25:
                continue

            bullets.append(ln)

        if not bullets:
            bullets = []
            for ln in lines:
                ln = re.sub(r'^[•–]\s+', '- ', ln)
                if not ln.startswith('- '): ln = f"- {ln}"
                if any(k in ln.lower() for k in self._instr_keywords): continue
                words = ln[2:].split()
                if len(words) > 18: ln = "- " + " ".join(words[:18])
                bullets.append(ln)
            if not bullets:
                return text.strip()

        bullets = bullets[:max_items] if len(bullets) >= min_items else bullets

        return "\n".join(bullets).strip()
    
    def _ngram_overlap(self, source, text, n = 6):
        def ngrams(s):
            toks = re.findall(r"\w+", s.lower())
            return {tuple(toks[i:i+n]) for i in range(len(toks)-n+1)}
        S, T = ngrams(source), ngrams(text)

        return 0.0 if not T else len(S & T) / max(1, len(T))


if __name__ == '__main__':
    character_updater = CharacterUpdater(file_path=r'Data\Character_Data_for_Chatbot\Killua_Zoldyck.json')
    character_updater.create_updated_character_data()