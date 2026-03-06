import logging
from typing import Any, Dict, List, Tuple
import re
from data_processor import DataProcessor
from db import Database


logger = logging.getLogger(__name__)


_processor = DataProcessor()
_db = Database()


def embed_question(question: str) -> List[float]:
    """
    Convert a user question into an embedding vector using the shared
    lightweight embedding model from DataProcessor.
    """
    if not question.strip():
        raise ValueError("Question must be a non-empty string.")

    embedding = _processor.embed_text(question)
    logger.debug("Generated question embedding of length %d", len(embedding))
    return embedding


def retrieve(
    question_embedding: List[float],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most similar chunks from the vector store.

    Returns a list of dictionaries with: id, text, source, score.
    """
    if not isinstance(question_embedding, list) or not question_embedding:
        raise ValueError("question_embedding must be a non-empty list of floats.")

    chunks = _db.get_similar_chunks(question_embedding, top_k=k)
    logger.info("Retrieved %d similar chunks (top_k=%d).", len(chunks), k)
    return chunks


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Build a cleaned context string from retrieved chunks.
    Removes words 'context' and 'answer' and unwanted symbols.
    """
    if not chunks:
        logger.warning("No chunks retrieved; using empty context.")
        return ""

    # Join all chunk texts
    context = "\n\n".join(chunk.get("text", "") for chunk in chunks if chunk.get("text"))

    # Remove the words 'context' and 'answer' (case-insensitive)
    context = re.sub(r'\b(context|answer)\b', '', context, flags=re.IGNORECASE)

    # Remove unwanted symbols: { } " '
    context = re.sub(r'[{}"\']', '', context)

    # Optionally, remove extra spaces caused by removal
    context = re.sub(r'\s+', ' ', context).strip()

    logger.debug("Built context of length %d characters.", len(context))
    return context


def build_qa_context(qa_items: List[Dict[str, Any]]) -> str:
    """
    Build a context string from (context, answer) rows in qa_context.

    Format:
        1.
        Context:
        ...

        Answer:
        ...
    """
    if not qa_items:
        logger.warning("No QA items retrieved; using empty context.")
        return ""

    parts: List[str] = []
    for idx, item in enumerate(qa_items, start=1):
        ctx = item.get("context", "")
        ans = item.get("answer", "")
        if not ctx and not ans:
            continue
        parts.append(
            f"{idx}.\nContext:\n{ctx}\n\nAnswer:\n{ans}"
        )

    context = "\n\n".join(parts)

    print(50*"-"+"  context  "+50*"-")
    print(context)
    print(50*"-"+"  context  "+50*"-")
    
    logger.debug("Built QA context of length %d characters.", len(context))
    return context

def run_rag_pipeline(
    question: str,
    top_k: int = 5,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Full RAG pipeline:

    1. Convert question to embedding.
    2. Retrieve most similar chunks from the vector store.
    3. Build textual context from chunks.
    4. Call ASSISTANT model with context + question.

    Returns:
        answer: Generated answer string.
        chunks: Retrieved chunks used to build the context.
    """
    logger.info("Running RAG pipeline. top_k=%d", top_k)

    query_embedding = embed_question(question)
    chunks = retrieve(query_embedding, k=top_k)
    context = build_context(chunks)
    print(50*"-")
    print(context)
    print(50*"-")
    answer = _processor.generate_answer(question=question, context=context)

    logger.info("Generated answer with length %d characters.", len(answer))
    return answer, chunks


def run_question_rag_pipeline(
    question: str,
    top_k: int = 1,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Alternative RAG pipeline that:

    1. Embeds the user question.
    2. Retrieves the top-k most similar stored questions
       from qa_questions (question table).
    3. Loads the corresponding context+answer rows from qa_context.
    4. Builds a combined context string from those rows.
    5. Calls ASSISTANT with that context + the original question.

    Returns:
        answer: generated answer string.
        similar_questions: rows from qa_questions.
        qa_items: rows from qa_context used to build the context.
    """
    logger.info("Running question-based RAG pipeline. top_k=%d", top_k)

    query_embedding = embed_question(question)
    similar_questions = _db.get_similar_questions(query_embedding, top_k=top_k)

    qa_ids = [row["qa_id"] for row in similar_questions]

    print(50*"+"+"  similar_questions  "+50*"+")
    print(similar_questions)
    print(50*"+"+"  similar_questions  "+50*"+")

    print(50*"+"+"  qa_ids  "+50*"+")
    print(qa_ids)
    print(50*"+"+"  qa_ids  "+50*"+")


    
    qa_items = _db.get_qa_contexts(qa_ids)

    context = build_qa_context(qa_items)


    print(50*"-"+"  context  "+50*"-")
    print(context)
    print(50*"-"+"  context  "+50*"-")
    answer = _processor.generate_answer(question=question, context=context)

    logger.info("Generated answer with length %d characters.", len(answer))
    return answer, similar_questions, qa_items


def generate_answer(question: str, top_k: int = 5) -> str:
    """
    Convenience wrapper for the full RAG pipeline.

    This matches the required high-level signature:

        answer = generate_answer(question: str) -> str
    """
    answer, _ = run_rag_pipeline(question=question, top_k=top_k)
    return answer

