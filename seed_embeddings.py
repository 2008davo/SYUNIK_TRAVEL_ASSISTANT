import os
import sys
import glob
import sqlite3
import torch

# ── Performance constraints ───────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_num_threads(1)

# ── Imports ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from db import Database
from data_processor import DataProcessor

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data", "FINAL")

print(f"[INFO] Data directory: {DATA_DIR}")

# ── Initialize services ───────────────────────────────────────────────────────
db = Database()
processor = DataProcessor()

print("[INFO] Database and DataProcessor initialized.")

# ── Clear existing chunks ─────────────────────────────────────────────────────
def clear_chunks():
    """Remove all existing chunk embeddings to prevent duplicates."""
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks")
        conn.commit()
        conn.close()

        print("[INFO] Existing chunks cleared.")

    except Exception as e:
        print(f"[ERROR] Failed clearing chunks: {e}")


# ── Seed from TXT files ───────────────────────────────────────────────────────
def seed_from_txt_files(txt_files):

    total_chunks = 0

    for filepath in txt_files:

        filename = os.path.basename(filepath)
        print(f"[INFO] Processing: {filename}")

        try:

            chunks = processor.chunk_file(
                filepath,
                chunk_size=500,
                overlap=50
            )

            # Clean chunks
            chunks = [str(c).strip() for c in chunks if str(c).strip()]

            if not chunks:
                print("  -> no valid chunks")
                continue

            # Batch embed
            embeddings = processor.embed_batch(chunks)

            for chunk, emb in zip(chunks, embeddings):
                db.save_chunk(chunk, emb, source=filename)

            total_chunks += len(chunks)

            print(f"  -> stored {len(chunks)} chunks")

        except Exception as e:
            print(f"[ERROR] Failed processing {filename}: {e}")

    print(f"\n[OK] Seeded {total_chunks} chunks from {len(txt_files)} file(s).")


# ── Built-in fallback knowledge ───────────────────────────────────────────────
def seed_builtin():

    SYUNIK_KNOWLEDGE = [

        ("Tatev Monastery is a 9th century Armenian Apostolic monastery located in Syunik Province on a large basalt plateau near the village of Tatev.", "tatev_monastery"),

        ("Wings of Tatev is the longest reversible aerial tramway in the world with a length of 5752 meters connecting Halidzor village to Tatev Monastery.", "wings_of_tatev"),

        ("Mount Khustup is a mountain near the city of Kapan in Syunik Province with an elevation of 3201 meters and is the burial site of Armenian national hero Garegin Nzhdeh.", "khustup"),

        ("Shaki Waterfall is an 18 meter high waterfall located near the town of Sisian in Syunik Province and is one of the most beautiful waterfalls in Armenia.", "shaki_waterfall"),

        ("Old Khndzoresk is a historical cave settlement in Syunik Province where people lived in caves until the middle of the twentieth century.", "khndzoresk"),

        ("Zorats Karer also known as Karahunj is a prehistoric megalithic monument near Sisian consisting of hundreds of standing stones.", "karahunj"),

        ("Meghri is a town in southern Syunik known for its warm subtropical climate and production of pomegranates figs and other fruits.", "meghri"),

        ("Goris is a historic town in Syunik famous for its unique rock formations called the Rock Forest and its nineteenth century city layout.", "goris"),

        ("Arevik National Park is located in southern Syunik and protects unique ecosystems and endangered species including the Caucasian leopard.", "arevik_national_park"),

        ("Ughtasar Petroglyphs are ancient rock carvings located at about 3300 meters altitude near Sisian and date back thousands of years.", "ughtasar_petroglyphs")

    ]

    texts = [x[0] for x in SYUNIK_KNOWLEDGE]
    sources = [x[1] for x in SYUNIK_KNOWLEDGE]

    embeddings = processor.embed_batch(texts)

    for text, emb, src in zip(texts, embeddings, sources):
        db.save_chunk(text, emb, source=src)

    print(f"[OK] Seeded {len(texts)} built-in knowledge chunks.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("Syunik Travel Assistant — Knowledge Base Seeder")
    print("=" * 60)

    # Initialize DB
    db.init_db()

    # Clear previous embeddings
    clear_chunks()

    # Find txt files
    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))

    if not txt_files:
        print("[WARNING] No TXT files found. Using fallback knowledge.")
        seed_builtin()
    else:
        print(f"[INFO] Found {len(txt_files)} txt file(s).\n")
        seed_from_txt_files(txt_files)

    print("\n[COMPLETE] RAG knowledge base ready.")
    print("Run: python app.py")