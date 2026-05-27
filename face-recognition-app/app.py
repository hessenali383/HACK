import os
import io
import json
import base64
import sqlite3
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DB_PATH = "database/faces.db"

# ── lazy-load heavy ML models ──────────────────────────────────────────────
_mtcnn = None
_model = None
_device = None

def get_models():
    global _mtcnn, _model, _device
    if _mtcnn is None:
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _mtcnn  = MTCNN(image_size=160, margin=20, device=_device, keep_all=True)
        _model  = InceptionResnetV1(pretrained="vggface2").eval().to(_device)
    return _mtcnn, _model, _device

# ── database helpers ───────────────────────────────────────────────────────
def init_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT    NOT NULL,
            embedding TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def load_database():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT name, embedding FROM faces").fetchall()
    conn.close()
    db = {}
    for name, emb_json in rows:
        db[name] = np.array(json.loads(emb_json))
    return db

def save_face_to_db(name: str, embedding: np.ndarray):
    conn = sqlite3.connect(DB_PATH)
    # Remove any previous entry with the same name so it stays fresh
    conn.execute("DELETE FROM faces WHERE name = ?", (name,))
    conn.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)",
                 (name, json.dumps(embedding.tolist())))
    conn.commit()
    conn.close()

# ── embedding helper ───────────────────────────────────────────────────────
def get_embedding_from_pil(pil_img):
    import torch
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    mtcnn, model, device = get_models()
    face_tensor = mtcnn(pil_img)
    if face_tensor is None:
        return None, None
    # keep_all=True may return a batch; take first
    if face_tensor.dim() == 3:
        face_tensor = face_tensor.unsqueeze(0)
    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        emb = model(face_tensor)
    return emb.cpu().numpy(), cos_sim  # return all embeddings

# ── routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/recognize", methods=["POST"])
def recognize():
    """
    Accepts a base64-encoded JPEG frame, detects all faces,
    returns list of {box, name, score} dicts.
    """
    try:
        data      = request.get_json(force=True)
        img_data  = data["image"].split(",")[-1]          # strip data:image/jpeg;base64,
        img_bytes = base64.b64decode(img_data)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        mtcnn, model, device = get_models()
        import torch
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        # Detect faces + boxes
        boxes, probs = mtcnn.detect(pil_img)
        if boxes is None:
            return jsonify({"faces": []})

        img_np  = np.array(pil_img)
        db      = load_database()
        results = []
        THRESHOLD = 0.55

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [max(0, int(c)) for c in box]
            face_crop = pil_img.crop((x1, y1, x2, y2)).resize((160, 160))

            face_tensor = mtcnn(face_crop)
            if face_tensor is None:
                name  = "Unknown"
                score = 0.0
            else:
                if face_tensor.dim() == 3:
                    face_tensor = face_tensor.unsqueeze(0)
                with torch.no_grad():
                    emb = model(face_tensor.to(device)).cpu().numpy()[0]

                name       = "Unknown"
                best_score = -1.0
                for pname, pemb in db.items():
                    s = cos_sim([emb], [pemb])[0][0]
                    if s > best_score:
                        best_score = s
                        if s >= THRESHOLD:
                            name = pname
                score = float(best_score)

            results.append({
                "box":   [x1, y1, x2, y2],
                "name":  name,
                "score": round(score, 3)
            })

        return jsonify({"faces": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/save-face", methods=["POST"])
def save_face():
    """
    Accepts {name, image (base64 JPEG), box [x1,y1,x2,y2]}.
    Crops the face, computes its embedding, and stores it.
    """
    try:
        data      = request.get_json(force=True)
        name      = data["name"].strip()
        img_data  = data["image"].split(",")[-1]
        img_bytes = base64.b64decode(img_data)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        box = data.get("box")
        if box:
            x1, y1, x2, y2 = [max(0, int(c)) for c in box]
            face_crop = pil_img.crop((x1, y1, x2, y2)).resize((160, 160))
        else:
            face_crop = pil_img

        mtcnn, model, device = get_models()
        import torch

        face_tensor = mtcnn(face_crop)
        if face_tensor is None:
            # fall back: try full image
            face_tensor = mtcnn(pil_img)
        if face_tensor is None:
            return jsonify({"success": False, "error": "No face detected"}), 400

        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        with torch.no_grad():
            emb = model(face_tensor.to(device)).cpu().numpy()[0]

        save_face_to_db(name, emb)
        return jsonify({"success": True, "name": name})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/faces", methods=["GET"])
def list_faces():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, name FROM faces").fetchall()
    conn.close()
    return jsonify({"faces": [{"id": r[0], "name": r[1]} for r in rows]})

@app.route("/api/faces/<int:face_id>", methods=["DELETE"])
def delete_face(face_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# ── startup ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
