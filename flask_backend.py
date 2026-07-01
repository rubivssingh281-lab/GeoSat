import os
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_file
from PIL import Image
from torchvision import transforms

from train import Encoder


app = Flask(__name__)

BASE_DIR = Path("data")
SAR_DIR = BASE_DIR / "SAR"
OPTICAL_DIR = BASE_DIR / "Optical"
PAIRED_DIR = BASE_DIR / "Paired_SAR_Optical_images"

SAVE_DIR = Path("saved_model")
SAR_MODEL = SAVE_DIR / "sar_encoder.pth"
OPTICAL_MODEL = SAVE_DIR / "optical_encoder.pth"
EMBEDDING_CACHE_PATH = SAVE_DIR / "optical_embeddings.pt"
EMBEDDING_FILES_LIST = SAVE_DIR / "optical_files.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 if DEVICE.type == "cuda" else 32

SAR_ENCODER = None
OPTICAL_ENCODER = None


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "GeoSat Backend API is Live",
        "health": "/api/health",
        "status_api": "/api/status"
    })


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


TRANSFORM_SAR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

TRANSFORM_OPTICAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def get_data_dirs():
    if SAR_DIR.is_dir() and OPTICAL_DIR.is_dir():
        return SAR_DIR, OPTICAL_DIR
    if PAIRED_DIR.is_dir():
        return PAIRED_DIR, PAIRED_DIR
    return None, None


def list_images(directory):
    if directory is None or not directory.is_dir():
        return []

    valid = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    return sorted([
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in valid
    ])


def load_models():
    global SAR_ENCODER, OPTICAL_ENCODER

    if SAR_ENCODER is not None and OPTICAL_ENCODER is not None:
        return SAR_ENCODER, OPTICAL_ENCODER

    if not SAR_MODEL.exists():
        raise FileNotFoundError(f"SAR model not found: {SAR_MODEL}")

    if not OPTICAL_MODEL.exists():
        raise FileNotFoundError(f"Optical model not found: {OPTICAL_MODEL}")

    sar_encoder = Encoder().to(DEVICE)
    optical_encoder = Encoder().to(DEVICE)

    sar_encoder.load_state_dict(torch.load(SAR_MODEL, map_location=DEVICE))
    optical_encoder.load_state_dict(torch.load(OPTICAL_MODEL, map_location=DEVICE))

    sar_encoder.eval()
    optical_encoder.eval()

    SAR_ENCODER = sar_encoder
    OPTICAL_ENCODER = optical_encoder

    return SAR_ENCODER, OPTICAL_ENCODER


def cache_is_valid(files):
    if not EMBEDDING_CACHE_PATH.exists() or not EMBEDDING_FILES_LIST.exists():
        return False

    cached_files = EMBEDDING_FILES_LIST.read_text(encoding="utf-8").splitlines()
    return cached_files == files


def build_optical_database(optical_encoder, optical_dir, files, limit=None, refresh_cache=False):
    selected_files = files[:limit] if limit else files

    if not refresh_cache and cache_is_valid(selected_files):
        return selected_files, torch.load(EMBEDDING_CACHE_PATH, map_location="cpu")

    embeddings = []

    with torch.no_grad():
        for start in range(0, len(selected_files), BATCH_SIZE):
            batch_files = selected_files[start:start + BATCH_SIZE]
            batch_images = []

            for filename in batch_files:
                img = Image.open(optical_dir / filename).convert("RGB")
                batch_images.append(TRANSFORM_OPTICAL(img))

            batch_tensor = torch.stack(batch_images).to(DEVICE)
            embeddings.append(optical_encoder(batch_tensor).cpu())

    embeddings = torch.cat(embeddings, dim=0)

    SAVE_DIR.mkdir(exist_ok=True)
    torch.save(embeddings, EMBEDDING_CACHE_PATH)
    EMBEDDING_FILES_LIST.write_text("\n".join(selected_files) + "\n", encoding="utf-8")

    return selected_files, embeddings


def score_to_percent(score):
    return max(0, min(100, round(float(score) * 100)))


def encode_query_image(sar_encoder, sar_dir, query_file=None, uploaded_file=None):
    if uploaded_file:
        img = Image.open(uploaded_file).convert("L")
        query_name = uploaded_file.filename or "uploaded_query"
    else:
        if not query_file:
            raise ValueError("Provide query_file or upload an image file.")

        image_path = sar_dir / query_file

        if not image_path.is_file():
            raise FileNotFoundError(f"Query image not found: {query_file}")

        img = Image.open(image_path).convert("L")
        query_name = query_file

    tensor = TRANSFORM_SAR(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = sar_encoder(tensor)

    return query_name, embedding


def retrieve_matches(query_file=None, uploaded_file=None, k=6, limit=2000, refresh_cache=False):
    sar_dir, optical_dir = get_data_dirs()

    if sar_dir is None or optical_dir is None:
        raise FileNotFoundError(
            "Dataset not found. Add data/SAR + data/Optical or data/Paired_SAR_Optical_images."
        )

    optical_files = list_images(optical_dir)

    if not optical_files:
        raise RuntimeError(f"No optical images found in {optical_dir}")

    sar_encoder, optical_encoder = load_models()

    query_name, query_embedding = encode_query_image(
        sar_encoder,
        sar_dir,
        query_file,
        uploaded_file
    )

    indexed_files, optical_embeddings = build_optical_database(
        optical_encoder,
        optical_dir,
        optical_files,
        limit=limit,
        refresh_cache=refresh_cache,
    )

    optical_embeddings = optical_embeddings.to(DEVICE)
    similarities = torch.matmul(query_embedding, optical_embeddings.T).squeeze(0)

    top_scores, top_indices = torch.topk(
        similarities,
        min(k, len(indexed_files))
    )

    results = []

    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), start=1):
        filename = indexed_files[int(idx)]

        results.append({
            "rank": rank,
            "filename": filename,
            "score": round(float(score), 6),
            "match_percent": score_to_percent(score),
            "image_url": f"/api/image/optical/{filename}",
        })

    return query_name, results


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": DEVICE.type,
    })


@app.route("/api/status", methods=["GET"])
def status():
    sar_dir, optical_dir = get_data_dirs()
    sar_files = list_images(sar_dir)
    optical_files = list_images(optical_dir)

    return jsonify({
        "device": DEVICE.type,
        "dataset_found": sar_dir is not None and optical_dir is not None,
        "sar_images": len(sar_files),
        "optical_images": len(optical_files),
        "sar_model_found": SAR_MODEL.exists(),
        "optical_model_found": OPTICAL_MODEL.exists(),
        "embedding_cache_found": EMBEDDING_CACHE_PATH.exists(),
        "database_connected": False,
    })


@app.route("/api/images/<kind>", methods=["GET"])
def images(kind):
    sar_dir, optical_dir = get_data_dirs()

    if kind == "sar":
        return jsonify({"kind": kind, "files": list_images(sar_dir)})

    if kind == "optical":
        return jsonify({"kind": kind, "files": list_images(optical_dir)})

    return jsonify({"error": "kind must be 'sar' or 'optical'"}), 400


@app.route("/api/image/<kind>/<filename>", methods=["GET"])
def image(kind, filename):
    sar_dir, optical_dir = get_data_dirs()

    if kind == "sar":
        directory = sar_dir
    elif kind == "optical":
        directory = optical_dir
    else:
        return jsonify({"error": "kind must be 'sar' or 'optical'"}), 400

    if filename not in list_images(directory):
        return jsonify({"error": "image not found"}), 404

    return send_file(directory / filename)


@app.route("/api/retrieve", methods=["POST", "OPTIONS"])
def retrieve():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    try:
        uploaded_file = request.files.get("file")

        if uploaded_file:
            payload = request.form.to_dict()
        else:
            payload = request.get_json(silent=True) or {}

        query_file = payload.get("query_file")
        k = int(payload.get("k", 6))
        limit = int(payload.get("limit", 2000))
        refresh_cache = str(payload.get("refresh_cache", "false")).lower() == "true"

        query_name, results = retrieve_matches(
            query_file=query_file,
            uploaded_file=uploaded_file,
            k=k,
            limit=limit,
            refresh_cache=refresh_cache,
        )

        return jsonify({
            "query": query_name,
            "count": len(results),
            "results": results,
            "database_connected": False,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)