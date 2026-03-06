
import logging
import os
import uuid
import json
from flask import Flask, g, jsonify, render_template, request, session

from data_processor import DataProcessor
from db import Database
from rag_pipeline import run_rag_pipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "syunik-secret-key-change-in-production")

# ── Shared processor (loaded once at startup) ─────────────────────────────────
processor = DataProcessor()


def get_db():
    if "db" not in g:
        g.db = Database()
    return g.db

# ── Routes ─────────────────────────────────────────────────────────

@app.route("/")
@app.route("/index")
def home():
    return render_template("index.html")


@app.route("/tourism")
def tourism():
    return render_template("tourism.html")


@app.route("/places")
def places():
    return render_template("places.html")


@app.route("/chat")
def chat_page():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("chat.html")


# ── Chat API ───────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat_api():
    """
    POST /chat  { "message": "..." }
    1. Embed the question with sentence-transformers.
    2. Retrieve top-10 similar chunks from SQLite vector store.
    3. Feed context + question to GPT-2 for answer generation.
    Returns { "reply": "..." }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("message", "").strip():
        return jsonify({"reply": "Please enter a valid message."}), 400

    user_message = data["message"].strip()
    db = get_db()

    logger.info("Received chat message: %s", user_message)

    # Run full RAG pipeline (embedding → retrieval → context → ASSISTANT)
    answer, chunks = run_rag_pipeline(user_message, top_k=5)

    # Persist conversation
    session_id = session.get("session_id", str(uuid.uuid4()))
    db.save_conversation(session_id, user_message, answer)

    return jsonify(
        {
            "reply": answer,
            "sources": [c["source"] for c in chunks if c.get("source")],
        }
    )


# ── Run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)