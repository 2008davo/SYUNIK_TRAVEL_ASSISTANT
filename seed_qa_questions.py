import json
from pathlib import Path

from data_processor import DataProcessor
from db import Database


def main() -> None:
    """
    Populate the qa_context and qa_questions tables from data/places.json.

    This assumes each item in places.json has:
        - "question"
        - "context"
        - "answere"  (note the existing key name)
    """
    base_dir = Path(__file__).resolve().parent
    places_path = base_dir / "data" / "places.json"

    data = json.loads(places_path.read_text(encoding="utf-8"))

    db = Database()
    processor = DataProcessor()

    # Ensure schema exists
    db.init_db()

    # Clear old QA rows to avoid duplicates
    db.clear_qa_tables()

    count = 0
    for item in data:
        question = str(item.get("question", "")).strip()
        context = str(item.get("context", "")).strip()
        answer = str(item.get("answere", "")).strip()

        if not question or not context or not answer:
            continue

        embedding = processor.embed_text(question)
        db.save_qa_pair(
            question_text=question,
            context=context,
            answer=answer,
            embedding=embedding,
        )
        count += 1

    print(f"Seeded {count} QA pairs into qa_context / qa_questions tables.")


if __name__ == "__main__":
    main()

