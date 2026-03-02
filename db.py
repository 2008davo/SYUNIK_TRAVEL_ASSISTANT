import json
import sqlite3
from typing import Any, Dict, List

import os

import numpy as np

DB_PATH = os.environ.get("DB_PATH", "rag_tourism.db")


class Database:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH

    # ── Connection ────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Schema & seed ─────────────────────────────────────────────────────────

    def init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    text      TEXT    NOT NULL,
                    embedding TEXT    NOT NULL,
                    source    TEXT    DEFAULT 'unknown',
                    created   TEXT    DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT    NOT NULL,
                    role       TEXT    NOT NULL,
                    content    TEXT    NOT NULL,
                    created    TEXT    DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS places (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    category    TEXT,
                    description TEXT,
                    location    TEXT,
                    rating      REAL DEFAULT 0.0,
                    image_url   TEXT
                );

                CREATE TABLE IF NOT EXISTS tourism_highlights (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    title       TEXT NOT NULL,
                    description TEXT,
                    icon        TEXT DEFAULT '🌍'
                );
            """)
            self._seed(conn)

    def _seed(self, conn: sqlite3.Connection):
        if conn.execute("SELECT COUNT(*) FROM places").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO places (name, category, description, location, rating, image_url) VALUES (?,?,?,?,?,?)",
                [
                    ("Tatev Monastery",   "Historical", "9th-century monastery perched above Vorotan Gorge.",          "Goris, Syunik",   4.9, "https://d31qtdfy11mjj9.cloudfront.net/places/1511524391185564485.jpg"),
                    ("Shaki Waterfall",   "Nature",     "18 m waterfall near Sisian, surrounded by basalt cliffs.",    "Sisian, Syunik",  4.7, "https://images.pexels.com/photos/417173/pexels-photo-417173.jpeg"),
                    ("Old Khndzoresk",    "Historical", "Ancient cave settlement with 160 m suspension bridge.",       "Goris, Syunik",   4.8, "https://images.unsplash.com/photo-1506744038136-46273834b3fb"),
                    ("Zorats Karer",      "Historical", "Prehistoric observatory stones – 'Armenian Stonehenge'.",     "Sisian, Syunik",  4.8, "https://images.pexels.com/photos/158607/circle-stone-megalith-henge-158607.jpeg"),
                    ("Khustup Mountain",  "Nature",     "High peak attracting hikers and mountaineers.",               "Kapan, Syunik",   4.6, "https://images.unsplash.com/photo-1511988617509-a57c8a288659"),
                    ("Halidzor Fortress", "Historical", "Historic fortress with dramatic Zangezur mountain views.",    "Syunik",          4.5, "https://images.unsplash.com/photo-1529655683826-aba9b3e77383"),
                    ("Wings of Tatev",    "Attraction", "World's longest reversible aerial tramway (5.7 km).",         "Halidzor, Syunik",4.9, "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b"),
                    ("Arevik National Park","Nature",   "Biodiversity hotspot in southern Syunik near Meghri.",        "Meghri, Syunik",  4.7, "https://images.unsplash.com/photo-1501854140801-50d01698950b"),
                ],
            )

        if conn.execute("SELECT COUNT(*) FROM tourism_highlights").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO tourism_highlights (title, description, icon) VALUES (?,?,?)",
                [
                    ("Historic Monasteries", "Explore 9th-century stone monasteries perched on cliffs.",    "⛪"),
                    ("Nature & Hiking",      "Trek alpine meadows, gorges, and volcanic peaks.",            "🏔️"),
                    ("Ancient Sites",        "Discover prehistoric observatories and cave dwellings.",      "🗿"),
                    ("Local Culture",        "Taste Syunik cuisine, honey, and pomegranates.",              "🍽️"),
                ],
            )

        # Seed RAG knowledge base with Syunik tourism text
        if conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0:
            syunik_facts = [
                ("Tatev Monastery is a 9th-century Armenian Apostolic monastery located in the Syunik Province. "
                 "It sits dramatically above the Vorotan River gorge and is accessible via the Wings of Tatev, "
                 "the world's longest non-stop double track cable car at 5.7 km.", "seed"),
                ("The Wings of Tatev cable car connects the village of Halidzor to Tatev Monastery. "
                 "It opened in 2010 and holds the Guinness World Record for the longest reversible aerial tramway.", "seed"),
                ("Zorats Karer, also called Carahunge or the Armenian Stonehenge, is a prehistoric site near Sisian. "
                 "It consists of over 200 standing stones dating back 5,500 years, possibly used as an ancient observatory.", "seed"),
                ("Shaki Waterfall is an 18-metre waterfall located 6 km from Sisian city. "
                 "The waterfall flows over basalt rock formations and is especially powerful in spring.", "seed"),
                ("Old Khndzoresk is an ancient cave village near Goris. "
                 "It features hundreds of cave dwellings carved into rock faces and a 160-metre swinging suspension bridge.", "seed"),
                ("Goris is the cultural capital of Syunik Province, situated at 1,370 m elevation, 240 km from Yerevan. "
                 "It is known for its 19th-century grid-plan streets, rock formations called the Rock Forest, "
                 "and as the gateway to Tatev Monastery.", "seed"),
                ("Sisian is an ancient city in Syunik at 1,600 m elevation. "
                 "It is home to Zorats Karer, Shaki Waterfall, Vorotnavank Monastery, and the Ughtasar Petroglyphs.", "seed"),
                ("Meghri is the southernmost city of Armenia, situated at 600 m elevation on the Araks River. "
                 "It has a subtropical climate and is famous for pomegranates, figs, the Meghri Fortress, "
                 "and the Arevik National Park.", "seed"),
                ("Kajaran is Armenia's mining capital at 1,950 m elevation. "
                 "It is home to the Zangezur Copper-Molybdenum Combine. Nearby attractions include "
                 "Mount Kaputjugh (3,905 m) and the alpine Lake Kaputan.", "seed"),
                ("Khustup Mountain rises to 3,206 m near Kapan and is a popular hiking destination in Syunik. "
                 "The ascent typically takes 5–7 hours and offers panoramic views of the Zangezur range.", "seed"),
                ("Arevik National Park in southern Syunik near Meghri protects rare flora and fauna including "
                 "the Caucasian leopard, bezoar ibex, and Armenian mouflon. It covers over 34,000 hectares.", "seed"),
                ("Ughtasar Petroglyphs are 5,000-year-old rock carvings located on the slopes of Mount Ughtasar "
                 "near Sisian at 3,300 m elevation. They depict hunting scenes, animals, and astronomical symbols.", "seed"),
                ("The Vorotan Canyon near Tatev is one of the deepest gorges in Armenia. "
                 "It features the natural rock arch called Devil's Bridge and dramatic cliff scenery.", "seed"),
                ("Transport in Syunik: Goris to Yerevan takes 4–5 hours by minibus. "
                 "Sisian to Yerevan is 3.5–4 hours. Meghri to Yerevan takes 7–9 hours. "
                 "Tatev Monastery is 30 minutes from Goris by car.", "seed"),
                ("Best time to visit Syunik is May to October. "
                 "Summer (June–August) is ideal for hiking. Spring brings wildflowers and waterfalls. "
                 "Winters are cold and snowy, especially in higher-altitude towns like Kajaran and Dastakert.", "seed"),
            ]
            # We'll let app.py handle embedding these via /api/ingest or startup seeding
            # Store as plain text chunks without embeddings (init will embed them)
            for text, source in syunik_facts:
                conn.execute(
                    "INSERT INTO chunks (text, embedding, source) VALUES (?, ?, ?)",
                    (text, "[]", source),
                )

    # ── Chunks ────────────────────────────────────────────────────────────────

    def save_chunk(self, text: str, embedding: List[float], source: str = "unknown"):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chunks (text, embedding, source) VALUES (?, ?, ?)",
                (text, json.dumps(embedding), source),
            )

    def update_chunk_embedding(self, chunk_id: int, embedding: List[float]):
        with self._conn() as conn:
            conn.execute(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                (json.dumps(embedding), chunk_id),
            )

    def get_chunks_without_embeddings(self):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, text, source FROM chunks WHERE embedding = '[]'"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_similar_chunks(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Brute-force cosine similarity over all embedded chunks."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, text, embedding, source FROM chunks WHERE embedding != '[]'"
            ).fetchall()

        if not rows:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)

        scored = []
        for row in rows:
            try:
                vec = np.array(json.loads(row["embedding"]), dtype=np.float32)
                v_norm = np.linalg.norm(vec)
                sim = float(np.dot(q, vec) / (q_norm * v_norm)) if (q_norm * v_norm) else 0.0
            except Exception:
                sim = 0.0
            scored.append({"id": row["id"], "text": row["text"], "source": row["source"], "score": sim})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_chunk_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ── Conversations ─────────────────────────────────────────────────────────

    def save_conversation(self, session_id: str, user_message: str, assistant_reply: str):
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
                [
                    (session_id, "user", user_message),
                    (session_id, "assistant", assistant_reply),
                ],
            )

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, created FROM conversations WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Places ────────────────────────────────────────────────────────────────

    def get_all_places(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM places ORDER BY rating DESC").fetchall()
        return [dict(r) for r in rows]

    def get_places_by_category(self, category: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM places WHERE category = ? ORDER BY rating DESC", (category,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Tourism highlights ────────────────────────────────────────────────────

    def get_tourism_highlights(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM tourism_highlights").fetchall()
        return [dict(r) for r in rows]
