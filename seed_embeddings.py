import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
import sys
import glob
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db import Database
from data_processor import DataProcessor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/FINAL")
print(f"[INFO] Data directory: '{DATA_DIR}'")

db = Database()
processor = DataProcessor()
print("[INFO] Database and DataProcessor initialized.")



def clear_chunks():
    """Remove all existing chunk embeddings."""
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    conn.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()
    print("[INFO] Cleared existing chunks.")


def seed_from_txt_files(txt_files):
    """Chunk, embed, and store all .txt files from data/."""
    total_chunks = 0
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        print(f"[INFO] Processing '{filename}' ...", end=" ", flush=True)
        try:
            chunks = processor.chunk_file(filepath, chunk_size=300, overlap=50)
            for chunk in chunks:
                emb = processor.embed_text(chunk)
                db.save_chunk(chunk, emb, source=filename)
            print(f"{len(chunks)} chunks embedded.")
            total_chunks += len(chunks)
        except Exception as e:
            print(f"ERROR — {e}")

    print(f"\n[OK] Seeded {total_chunks} chunks from {len(txt_files)} file(s).")


def seed_builtin():
    """Fallback: seed hardcoded Syunik knowledge base."""
    SYUNIK_KNOWLEDGE = [
        ("Tatev Monastery is a 9th-century Armenian monastery on a basalt promontory above "
         "the Vorotan Gorge in Syunik Province. It was an important center of learning and "
         "science in medieval Armenia.", "tatev"),

        ("The Wings of Tatev is the world's longest non-stop double track cable car at 5.7 km. "
         "It holds a Guinness World Record and connects Halidzor to Tatev Monastery, "
         "with breathtaking views during the ~11-minute ride.", "tatev"),

        ("Zorats Karer (Karahunj / Armenian Stonehenge) near Sisian dates to ~5,500 years ago. "
         "Over 200 standing stones are arranged in a circular pattern believed to be "
         "an ancient astronomical observatory.", "sisian"),

        ("Shaki Waterfall is an 18-metre cascade near Sisian surrounded by basalt columns. "
         "Located about 6 km from the city, it is a popular day-trip destination.", "sisian"),

        ("Old Khndzoresk is a cave village in a deep gorge, inhabited until the 1950s. "
         "A famous 160-metre swinging suspension bridge lets visitors explore the site.", "khndzoresk"),

        ("Goris is the main city of southern Syunik at 1,370 m elevation, ~240 km from Yerevan. "
         "Known for its Rock Forest of pyramidal rock formations and as the gateway to Tatev.", "goris"),

        ("Sisian is a city in Syunik at 1,600 m. Attractions: Zorats Karer, Shaki Waterfall, "
         "Vorotnavank Monastery, and Ughtasar Petroglyphs. About 3.5–4 h from Yerevan.", "sisian"),

        ("Meghri is Armenia's southernmost city at 600 m near Iran, famous for pomegranates, "
         "figs, and olives. Key sights: Meghri Fortress, Arevik National Park. ~7–9 h from Yerevan.", "meghri"),

        ("Kajaran is a mining city at ~1,950 m. The ZCMC open-pit mine is one of the world's largest. "
         "Nearby: Mount Kaputjugh (3,905 m, highest in Syunik) and Lake Kaputan.", "kajaran"),

        ("Khustup Mountain rises to 3,206 m in the Zangezur range. Ascent takes 5–7 h, "
         "with panoramic views across southern Armenia and into Iran.", "hiking"),

        ("Arevik National Park covers 34,000 hectares in southern Syunik and is home to "
         "the Caucasian Leopard, bezoar ibex, Armenian mouflon, and hundreds of plant species.", "nature"),

        ("Ughtasar Petroglyphs are 5,000-year-old rock carvings at 3,300 m above Sisian, "
         "depicting hunting scenes, animals, and geometric patterns. Accessible only in summer.", "sisian"),

        ("Vorotan Canyon is the dramatic gorge of the Vorotan River, containing Devil's Bridge "
         "(a natural basalt arch) and forming the setting for Tatev Monastery.", "goris"),

        ("Getting to Syunik from Yerevan: Goris ~4–5 h, Sisian ~3.5–4 h, "
         "Meghri and Agarak 7–9 h. Tatev is 30 min from Goris by car.", "transport"),

        ("Best time to visit Syunik: May–October. Summer for hiking Khustup and Ughtasar. "
         "Spring for wildflowers and waterfalls. Tatev and Zorats Karer are year-round.", "general"),

        ("Agarak is Armenia's southernmost town on the Iranian border at ~600 m elevation. "
         "Founded in 1949 for the Agarak Copper-Molybdenum Combine, ~380 km from Yerevan. "
         "Known as the 'Southern Gate' for Armenia-Iran trade.", "agarak"),

        ("Agarak's subtropical climate yields figs, olives, pomegranates, and persimmons. "
         "Nearby: Arevik National Park, Karchevan village, Shvanidzor's 17th-c. aqueduct. "
         "Minibuses to Yerevan depart 07:00–08:00; journey takes 7–9 h.", "agarak"),
    ]

    for text, source in SYUNIK_KNOWLEDGE:
        emb = processor.embed_text(text)
        db.save_chunk(text, emb, source=source)

    print(f"[OK] Seeded {len(SYUNIK_KNOWLEDGE)} built-in Syunik knowledge chunks.")


if __name__ == "__main__":
    print("=" * 60)
    print("Syunik Travel Assistant — Knowledge Base Seeder")
    print("=" * 60)

    db.init_db()
    clear_chunks()

    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    print(f"[INFO] Found {len(txt_files)} .txt file(s) in data/\n")
    seed_from_txt_files(txt_files)

    print("\nDone! Run: python app.py")