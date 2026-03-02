import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import re
import textwrap
import hashlib
from typing import List
import numpy as np

# ── Simple TF-IDF style embedding (no external models, no threading issues) ───

VOCAB_SIZE = 384  # output vector size


def _simple_embed(text: str) -> List[float]:
    """
    Lightweight deterministic embedding using character n-gram hashing.
    No external libraries, no threads, no macOS mutex issues.
    Fast and good enough for small tourism knowledge bases.
    """

    text = text.lower().strip()
    vec = np.zeros(VOCAB_SIZE, dtype=np.float32)

    # Character trigrams
    for i in range(len(text) - 2):
        trigram = text[i:i + 3]
        idx = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % VOCAB_SIZE
        vec[idx] += 1.0

    # Word unigrams
    words = re.findall(r"\w+", text)
    for word in words:
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % VOCAB_SIZE
        vec[idx] += 2.0  # words weighted more than trigrams

    # Word bigrams
    for i in range(len(words) - 1):
        bigram = words[i] + "_" + words[i + 1]
        idx = int(hashlib.md5(bigram.encode()).hexdigest(), 16) % VOCAB_SIZE
        vec[idx] += 1.5

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec.tolist()


# ── Lazy GPT-2 loader ─────────────────────────────────────────────────────────

_gpt2_tokenizer = None
_gpt2_model = None
_device = None


def _get_gpt2():
    global _gpt2_tokenizer, _gpt2_model, _device

    if _gpt2_model is None:
        try:
            import torch
            torch.set_num_threads(1)
            from transformers import GPT2Tokenizer, GPT2LMHeadModel

            model_path = os.getenv("GPT2_MODEL_PATH", "gpt2")
            print(f"[INFO] Loading GPT-2 from '{model_path}'...")

            _gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            _gpt2_tokenizer.pad_token = _gpt2_tokenizer.eos_token

            _gpt2_model = GPT2LMHeadModel.from_pretrained(model_path)

            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _gpt2_model.to(_device)
            _gpt2_model.eval()

            print(f"[INFO] GPT-2 loaded from '{model_path}' on {_device}.")

        except Exception as e:
            _gpt2_model = "unavailable"
            print(f"[WARN] GPT-2 could not be loaded: {e}")

    return (
        (_gpt2_tokenizer, _gpt2_model, _device)
        if _gpt2_model != "unavailable"
        else (None, None, None)
    )


class DataProcessor:

    # ── Chunking ──────────────────────────────────────────────────────────────

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current = ""

        for sent in sentences:
            sub_parts = textwrap.wrap(sent, width=chunk_size) if len(sent) > chunk_size else [sent]

            for part in sub_parts:
                if len(current) + len(part) + 1 <= chunk_size:
                    current = (current + " " + part).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = part

        if current:
            chunks.append(current)

        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                tail = chunks[i - 1][-overlap:]
                overlapped.append((tail + " " + chunks[i]).strip())
            return overlapped

        return chunks

    def chunk_file(self, filepath: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> List[float]:
        return _simple_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [_simple_embed(t) for t in texts]

    # ── Generation ────────────────────────────────────────────────────────────

    def generate_answer(self, question: str, context: str) -> str:
        tokenizer, model, device = _get_gpt2()

        if model:
            print(f"[INFO] Generating answer with GPT-2 for question: '{question}'")
            return self._generate_gpt2(question, context, tokenizer, model, device)

        print(f"[INFO] Generating answer with template for question: '{question}'")
        return self._generate_template(question, context)

    def _generate_gpt2(self, question, context, tokenizer, model, device, max_length=128):
        try:
            import torch

            context_snippet = context[:500] if context else ""

            input_text = f"Context: {context_snippet}\nQuestion: {question}\nAnswer:"

            encoded = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=150,
                    min_new_tokens=20,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.9,
                    num_return_sequences=1,
                )

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text.strip()

            answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

            parts = answer.split(".")
            seen, deduped = set(), []

            for p in parts:
                key = p.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(p)

            answer = ".".join(deduped).strip()

            print(f"[INFO] Generated answer: '{answer}'")

            return answer if answer else self._generate_template(question, context)

        except Exception as e:
            print(f"[ERROR] GPT-2 generation failed: {e}")
            return self._generate_template(question, context)

    def _generate_template(self, question: str, context: str) -> str:
        if not context.strip():
            return (
                "I don't have enough information to answer that question. "
                "Please try rephrasing or ask about a different topic in Syunik."
            )

        q_words = set(re.findall(r"\w+", question.lower()))
        sentences = re.split(r"(?<=[.!?])\s+", context)

        best = context[:300]
        best_score = 0

        for sent in sentences:
            score = len(q_words & set(re.findall(r"\w+", sent.lower())))
            if score > best_score:
                best_score = score
                best = sent

        return f"Based on available information: {best}"

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)

        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0