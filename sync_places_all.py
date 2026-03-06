import json
from pathlib import Path


def main() -> None:
    """
    Sync `data/places.json` into `data/FINAL/ALL.txt`.

    - Ensures every (context, answere, question) triple from places.json
      is present in ALL.txt.
    - Rebuilds the Q&A JSON array section, removing duplicate entries.
    - Preserves the descriptive text that appears before the first '[' in ALL.txt.
    """
    base_dir = Path(__file__).resolve().parent
    places_path = base_dir / "data" / "places.json"
    all_path = base_dir / "data" / "FINAL" / "ALL.txt"

    data = json.loads(places_path.read_text(encoding="utf-8"))

    # De-duplicate by (context, question, answere)
    seen = set()
    unique_items = []
    for item in data:
        key = (
            item.get("context", "").strip(),
            item.get("question", "").strip(),
            item.get("answere", "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)

    qna_json = json.dumps(unique_items, ensure_ascii=False, indent=2)

    all_text = all_path.read_text(encoding="utf-8")
    idx = all_text.find("[")

    if idx == -1:
        prefix = all_text.rstrip() + "\n\n"
    else:
        prefix = all_text[:idx].rstrip() + "\n\n"

    all_path.write_text(prefix + qna_json + "\n", encoding="utf-8")
    print(f"Updated {all_path} with {len(unique_items)} unique Q&A entries.")


if __name__ == "__main__":
    main()

