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
from AI.model import Assistant

ASSISTANT = Assistant("lora_gpt2_my_v2")

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
        return ASSISTANT.answer_question(question=question, context=context)

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)

        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0