import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_file
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from train import Encoder, get_device


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("geosat.backend")

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "data"
SAR_DIR = BASE_DIR / "SAR"
OPTICAL_DIR = BASE_DIR / "Optical"
PAIRED_DIR = BASE_DIR / "Paired_SAR_Optical_images"

SAVE_DIR = PROJECT_DIR / "saved_model"
SAR_MODEL = SAVE_DIR / "sar_encoder.pth"
OPTICAL_MODEL = SAVE_DIR / "optical_encoder.pth"
EMBEDDING_CACHE_PATH = SAVE_DIR / "optical_embeddings.pt"
EMBEDDING_FILES_LIST = SAVE_DIR / "optical_files.txt"
CACHE_METADATA_PATH = SAVE_DIR / "optical_cache.json"

EMBEDDING_DIM = 256
DEFAULT_TOP_K = 6
MAX_TOP_K = 100
DEFAULT_LIMIT = 2000
VALID_IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
})

MAX_UPLOAD_MB = max(1, int(os.environ.get("GEOSAT_MAX_UPLOAD_MB", 25)))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
CORS_ALLOW_ORIGIN = os.environ.get("GEOSAT_CORS_ORIGINS", "*")

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

DEVICE = get_device()
BATCH_SIZE = 64 if DEVICE.type == "cuda" else 32

_MODEL_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()
_METRICS_LOCK = threading.Lock()

SAR_ENCODER: Encoder | None = None
OPTICAL_ENCODER: Encoder | None = None
_OPTICAL_CACHE: dict[str, Any] = {
    "optical_dir": None,
    "files": None,
    "embeddings": None,
    "model_signature": None,
}

QUERY_METRICS: dict[str, Any] = {
    "total_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "recent_latency_ms": deque(maxlen=50),
}


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


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "GeoSat Backend API is Live",
        "health": "/api/health",
        "status_api": "/api/status",
    })


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = CORS_ALLOW_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.errorhandler(404)
def handle_not_found(_exc):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def handle_method_not_allowed(_exc):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(413)
def handle_payload_too_large(_exc):
    return jsonify({
        "error": f"Uploaded file exceeds maximum allowed size of {MAX_UPLOAD_MB} MB."
    }), 413


def _record_query_result(success: bool, latency_ms: float | None = None) -> None:
    with _METRICS_LOCK:
        QUERY_METRICS["total_queries"] += 1
        if success:
            QUERY_METRICS["successful_queries"] += 1
            if latency_ms is not None:
                QUERY_METRICS["recent_latency_ms"].append(latency_ms)
        else:
            QUERY_METRICS["failed_queries"] += 1


def metrics_snapshot() -> dict[str, Any]:
    with _METRICS_LOCK:
        latencies = list(QUERY_METRICS["recent_latency_ms"])
        total = QUERY_METRICS["total_queries"]
        successful = QUERY_METRICS["successful_queries"]
        failed = QUERY_METRICS["failed_queries"]

    return {
        "total_queries": total,
        "successful_queries": successful,
        "failed_queries": failed,
        "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "last_latency_ms": latencies[-1] if latencies else None,
    }


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _parse_int(value: Any, name: str, min_value: int | None = None, max_value: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"{name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer.")
    if min_value is not None and parsed < min_value:
        if max_value is not None:
            raise ValueError(f"{name} must be between {min_value} and {max_value}.")
        raise ValueError(f"{name} must be at least {min_value}.")
    if max_value is not None and parsed > max_value:
        if min_value is not None:
            raise ValueError(f"{name} must be between {min_value} and {max_value}.")
        raise ValueError(f"{name} must be at most {max_value}.")
    return parsed


def _parse_limit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"none", "all"}:
            return None
        if not cleaned:
            raise ValueError("limit must be at least 1, 'none', or 'all'.")
    return _parse_int(value, "limit", min_value=1)


def _parse_bool(value: Any, name: str = "refresh_cache") -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "1", "yes"}:
            return True
        if cleaned in {"false", "0", "no"}:
            return False
    raise ValueError(f"{name} must be a boolean ('true' or 'false').")


def get_data_dirs() -> tuple[Path | None, Path | None]:
    if SAR_DIR.is_dir() and OPTICAL_DIR.is_dir():
        return SAR_DIR, OPTICAL_DIR
    if PAIRED_DIR.is_dir():
        return PAIRED_DIR, PAIRED_DIR
    return None, None


def list_images(directory: Path | None) -> list[str]:
    if directory is None or not directory.is_dir():
        return []

    return sorted(
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


# Security: Defend against PIL decompression bombs (max 25 Megapixels, approx 5000x5000)
Image.MAX_IMAGE_PIXELS = 25_000_000

# Magic bytes signatures for supported image types
MAGIC_BYTES_SIGNATURES: tuple[bytes, ...] = (
    b"\x89PNG\r\n\x1a\n",             # PNG
    b"\xff\xd8\xff",                   # JPEG
    b"BM",                             # BMP
    b"II*\x00",                        # TIFF (little endian)
    b"MM\x00*",                        # TIFF (big endian)
    b"RIFF",                           # WebP header prefix
)


def validate_image_magic_bytes(header: bytes) -> bool:
    """Validate raw binary stream header against known legitimate image signatures."""
    if not header or len(header) < 2:
        return False
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if header.startswith(b"\xff\xd8\xff"):
        return True
    if header.startswith(b"BM"):
        return True
    if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
        return True
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return True
    return False


def resolve_safe_image_path(directory: Path | None, filename: str | None) -> Path | None:
    if directory is None or not filename or not isinstance(filename, str):
        return None
    # Defense against null bytes and control characters
    if "\x00" in filename or "\r" in filename or "\n" in filename:
        return None
    if Path(filename).name != filename:
        return None

    resolved_dir = directory.resolve()
    image_path = (resolved_dir / filename).resolve()

    try:
        # Commonpath verification prevents directory traversal via symlinks or Windows junctions
        if os.path.commonpath([str(image_path), str(resolved_dir)]) != str(resolved_dir):
            return None
    except ValueError:
        # Raised on Windows when paths are on different drive letters
        return None

    if (
        image_path.parent != resolved_dir
        or not image_path.is_file()
        or image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS
    ):
        return None
    return image_path


def load_checkpoint(path: Path, map_location: Any = None):
    target_device = DEVICE if map_location is None else map_location
    try:
        return torch.load(path, map_location=target_device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=target_device)


def load_models() -> tuple[Encoder, Encoder]:
    global SAR_ENCODER, OPTICAL_ENCODER

    if SAR_ENCODER is not None and OPTICAL_ENCODER is not None:
        return SAR_ENCODER, OPTICAL_ENCODER

    with _MODEL_LOCK:
        if SAR_ENCODER is not None and OPTICAL_ENCODER is not None:
            return SAR_ENCODER, OPTICAL_ENCODER

        if not SAR_MODEL.exists():
            raise FileNotFoundError(f"SAR model not found: {SAR_MODEL}")
        if not OPTICAL_MODEL.exists():
            raise FileNotFoundError(f"Optical model not found: {OPTICAL_MODEL}")

        started_at = time.perf_counter()
        sar_encoder = Encoder(embedding_dim=EMBEDDING_DIM).to(DEVICE)
        optical_encoder = Encoder(embedding_dim=EMBEDDING_DIM).to(DEVICE)

        sar_encoder.load_state_dict(load_checkpoint(SAR_MODEL))
        optical_encoder.load_state_dict(load_checkpoint(OPTICAL_MODEL))

        sar_encoder.eval()
        optical_encoder.eval()

        SAR_ENCODER = sar_encoder
        OPTICAL_ENCODER = optical_encoder

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info("Loaded SAR and optical encoders on %s in %.2f ms", DEVICE.type, elapsed_ms)
        return SAR_ENCODER, OPTICAL_ENCODER


def _read_disk_cache_manifest(
    files: list[str],
    optical_dir: Path,
) -> list[str] | None:
    if (
        not EMBEDDING_CACHE_PATH.exists()
        or not EMBEDDING_FILES_LIST.exists()
        or not CACHE_METADATA_PATH.exists()
    ):
        return None

    try:
        cached_files = [
            line for line in EMBEDDING_FILES_LIST.read_text(encoding="utf-8").splitlines()
            if line
        ]
        metadata = json.loads(CACHE_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if metadata.get("optical_dir") != str(optical_dir.resolve()):
        return None
    if metadata.get("embedding_dimension") != EMBEDDING_DIM:
        return None

    model_sig = _file_signature(OPTICAL_MODEL)
    recorded_mtime = metadata.get("optical_model_mtime_ns")
    recorded_size = metadata.get("optical_model_size")
    if model_sig is not None and recorded_mtime is not None and recorded_mtime != model_sig[0]:
        return None
    if model_sig is not None and recorded_size is not None and recorded_size != model_sig[1]:
        return None

    if len(cached_files) >= len(files) and cached_files[:len(files)] == files:
        return cached_files
    return None


def cache_is_valid(files: list[str], optical_dir: Path) -> bool:
    return _read_disk_cache_manifest(files, optical_dir) is not None


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(tensor, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _set_memory_cache(
    optical_dir: Path,
    files: list[str],
    embeddings: torch.Tensor,
    model_sig: tuple[int, int] | None,
) -> torch.Tensor:
    device_embeddings = F.normalize(embeddings.to(device=DEVICE, dtype=torch.float32), dim=1)
    _OPTICAL_CACHE["optical_dir"] = str(optical_dir.resolve())
    _OPTICAL_CACHE["files"] = list(files)
    _OPTICAL_CACHE["embeddings"] = device_embeddings
    _OPTICAL_CACHE["model_signature"] = model_sig
    return device_embeddings


def _get_memory_cache(
    optical_dir: Path,
    selected_files: list[str],
    model_sig: tuple[int, int] | None,
) -> tuple[list[str], torch.Tensor] | None:
    cached_files = _OPTICAL_CACHE["files"]
    cached_embeddings = _OPTICAL_CACHE["embeddings"]
    if (
        cached_files is None
        or cached_embeddings is None
        or _OPTICAL_CACHE["optical_dir"] != str(optical_dir.resolve())
        or _OPTICAL_CACHE["model_signature"] != model_sig
    ):
        return None
    if len(cached_files) >= len(selected_files) and cached_files[:len(selected_files)] == selected_files:
        return selected_files, cached_embeddings[:len(selected_files)]
    return None


def _is_memory_cache_loaded() -> bool:
    return _OPTICAL_CACHE["embeddings"] is not None


def clear_memory_cache() -> None:
    with _CACHE_LOCK:
        _OPTICAL_CACHE["optical_dir"] = None
        _OPTICAL_CACHE["files"] = None
        _OPTICAL_CACHE["embeddings"] = None
        _OPTICAL_CACHE["model_signature"] = None


def build_optical_database(
    optical_encoder: Encoder,
    optical_dir: Path,
    files: list[str],
    limit: int | None = None,
    refresh_cache: bool = False,
) -> tuple[list[str], torch.Tensor]:
    selected_files = files[:limit] if limit is not None else list(files)
    if not selected_files:
        raise RuntimeError(f"No optical images available to index in {optical_dir}")

    model_sig = _file_signature(OPTICAL_MODEL)

    with _CACHE_LOCK:
        if not refresh_cache:
            memory_hit = _get_memory_cache(optical_dir, selected_files, model_sig)
            if memory_hit is not None:
                return memory_hit

            cached_files = _read_disk_cache_manifest(selected_files, optical_dir)
            if cached_files is not None:
                try:
                    disk_embeddings = load_checkpoint(EMBEDDING_CACHE_PATH, map_location="cpu")
                except Exception:
                    logger.warning("Failed to load cached embeddings from disk; rebuilding cache.")
                    disk_embeddings = None

                if (
                    isinstance(disk_embeddings, torch.Tensor)
                    and disk_embeddings.ndim == 2
                    and disk_embeddings.shape[0] == len(cached_files)
                    and disk_embeddings.shape[1] == EMBEDDING_DIM
                ):
                    logger.info(
                        "Loaded %d optical embeddings from disk cache into %s memory",
                        len(cached_files),
                        DEVICE.type,
                    )
                    device_embeddings = _set_memory_cache(
                        optical_dir, cached_files, disk_embeddings, model_sig
                    )
                    return selected_files, device_embeddings[:len(selected_files)]

        logger.info("Computing optical embeddings for %d images...", len(selected_files))
        started_at = time.perf_counter()
        embedding_batches: list[torch.Tensor] = []

        with torch.inference_mode():
            for start in range(0, len(selected_files), BATCH_SIZE):
                batch_files = selected_files[start:start + BATCH_SIZE]
                batch_images = []

                for filename in batch_files:
                    image_path = optical_dir / filename
                    try:
                        with Image.open(image_path) as img:
                            batch_images.append(TRANSFORM_OPTICAL(img.convert("RGB")))
                    except (OSError, UnidentifiedImageError) as exc:
                        raise RuntimeError(f"Failed to read optical image '{filename}': {exc}") from exc

                batch_tensor = torch.stack(batch_images).to(DEVICE)
                embedding_batches.append(optical_encoder(batch_tensor).cpu())

        cpu_embeddings = torch.cat(embedding_batches, dim=0)

        metadata = {
            "optical_dir": str(optical_dir.resolve()),
            "embedding_dimension": int(cpu_embeddings.shape[1]),
            "count": len(selected_files),
            "optical_model_mtime_ns": model_sig[0] if model_sig is not None else None,
            "optical_model_size": model_sig[1] if model_sig is not None else None,
        }

        _atomic_save_tensor(EMBEDDING_CACHE_PATH, cpu_embeddings)
        _atomic_write_text(EMBEDDING_FILES_LIST, "\n".join(selected_files) + "\n")
        _atomic_write_text(CACHE_METADATA_PATH, json.dumps(metadata, indent=2) + "\n")

        device_embeddings = _set_memory_cache(optical_dir, selected_files, cpu_embeddings, model_sig)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info("Built and cached %d optical embeddings in %.2f ms", len(selected_files), elapsed_ms)

        return selected_files, device_embeddings


def score_to_percent(score: float | torch.Tensor) -> int:
    return max(0, min(100, round(float(score) * 100)))


def encode_query_image(
    sar_encoder: Encoder,
    sar_dir: Path,
    query_file: str | None = None,
    uploaded_file: Any = None,
) -> tuple[str, torch.Tensor]:
    if uploaded_file is not None:
        raw_name = getattr(uploaded_file, "filename", None) or "uploaded_query"
        query_name = Path(raw_name.replace("\\", "/")).name or "uploaded_query"
        if "\x00" in query_name:
            raise ValueError("Uploaded filename contains illegal null byte characters.")

        stream = getattr(uploaded_file, "stream", uploaded_file)
        if hasattr(stream, "seek") and hasattr(stream, "read"):
            header = stream.read(16)
            stream.seek(0)
            if not validate_image_magic_bytes(header):
                raise ValueError("Uploaded file does not match a valid image magic byte signature.")

        try:
            with Image.open(uploaded_file) as img:
                tensor = TRANSFORM_SAR(img.convert("L")).unsqueeze(0).to(DEVICE)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
            raise ValueError("Uploaded file is not a valid or readable image.") from exc
    else:
        if not query_file:
            raise ValueError("Provide query_file or upload an image file.")

        image_path = resolve_safe_image_path(sar_dir, query_file)
        if image_path is None:
            raise FileNotFoundError(f"Query image not found: {query_file}")

        try:
            with Image.open(image_path) as img:
                tensor = TRANSFORM_SAR(img.convert("L")).unsqueeze(0).to(DEVICE)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
            raise ValueError(f"Query image '{query_file}' could not be decoded.") from exc
        query_name = query_file

    with torch.inference_mode():
        embedding = F.normalize(sar_encoder(tensor).to(dtype=torch.float32), dim=1)

    return query_name, embedding


def retrieve_matches(
    query_file: str | None = None,
    uploaded_file: Any = None,
    k: int = DEFAULT_TOP_K,
    limit: int | None = DEFAULT_LIMIT,
    refresh_cache: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    k = _parse_int(k, "k", min_value=1, max_value=MAX_TOP_K)
    limit = _parse_limit(limit)

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
        query_file=query_file,
        uploaded_file=uploaded_file,
    )

    indexed_files, optical_embeddings = build_optical_database(
        optical_encoder,
        optical_dir,
        optical_files,
        limit=limit,
        refresh_cache=refresh_cache,
    )

    with torch.inference_mode():
        similarities = torch.matmul(
            query_embedding,
            optical_embeddings.to(device=DEVICE, dtype=query_embedding.dtype).T,
        ).squeeze(0)
        top_k = min(k, len(indexed_files))
        top_scores, top_indices = torch.topk(similarities, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
        filename = indexed_files[idx]
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
        "models_loaded": SAR_ENCODER is not None and OPTICAL_ENCODER is not None,
        "cache_loaded_in_memory": _is_memory_cache_loaded(),
        "metrics": metrics_snapshot(),
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    return jsonify(metrics_snapshot())


@app.route("/api/status", methods=["GET"])
def status():
    sar_dir, optical_dir = get_data_dirs()
    sar_files = list_images(sar_dir)
    optical_files = list_images(optical_dir)
    cache_in_memory = _is_memory_cache_loaded()

    return jsonify({
        "device": DEVICE.type,
        "dataset_found": sar_dir is not None and optical_dir is not None,
        "sar_images": len(sar_files),
        "optical_images": len(optical_files),
        "sar_model_found": SAR_MODEL.exists(),
        "optical_model_found": OPTICAL_MODEL.exists(),
        "models_loaded": SAR_ENCODER is not None and OPTICAL_ENCODER is not None,
        "embedding_cache_found": EMBEDDING_CACHE_PATH.exists(),
        "embedding_cache_metadata_found": CACHE_METADATA_PATH.exists(),
        "cache_loaded_in_memory": cache_in_memory,
        "database_connected": bool(optical_files and (EMBEDDING_CACHE_PATH.exists() or cache_in_memory)),
        "embedding_dimension": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
        "framework": "PyTorch",
    })


@app.route("/api/images/<kind>", methods=["GET"])
def images(kind: str):
    sar_dir, optical_dir = get_data_dirs()

    if kind == "sar":
        all_files = list_images(sar_dir)
    elif kind == "optical":
        all_files = list_images(optical_dir)
    else:
        return jsonify({"error": "kind must be 'sar' or 'optical'"}), 400

    try:
        offset_arg = request.args.get("offset")
        limit_arg = request.args.get("limit")
        offset = _parse_int(offset_arg, "offset", min_value=0) if offset_arg is not None else 0
        limit = _parse_int(limit_arg, "limit", min_value=1) if limit_arg is not None else None
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    sliced = all_files[offset:offset + limit] if limit is not None else all_files[offset:]
    return jsonify({
        "kind": kind,
        "total": len(all_files),
        "files": sliced,
    })


@app.route("/api/image/<kind>/<filename>", methods=["GET"])
def image(kind: str, filename: str):
    sar_dir, optical_dir = get_data_dirs()

    if kind == "sar":
        directory = sar_dir
    elif kind == "optical":
        directory = optical_dir
    else:
        return jsonify({"error": "kind must be 'sar' or 'optical'"}), 400

    image_path = resolve_safe_image_path(directory, filename)
    if image_path is None:
        return jsonify({"error": "image not found"}), 404

    return send_file(image_path, conditional=True, max_age=3600)


@app.route("/api/retrieve", methods=["POST", "OPTIONS"])
def retrieve():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    # Parse and validate inputs before starting query execution/metrics.
    try:
        uploaded_file = request.files.get("file")

        if uploaded_file is not None:
            payload: dict[str, Any] = request.form.to_dict()
        else:
            if request.is_json:
                parsed_json = request.get_json(silent=True)
                if parsed_json is None:
                    raise ValueError("Invalid JSON request body.")
                if not isinstance(parsed_json, dict):
                    raise ValueError("JSON request body must be an object.")
                payload = parsed_json
            else:
                payload = request.form.to_dict()

        query_file = payload.get("query_file")
        if uploaded_file is not None and not uploaded_file.filename:
            raise ValueError("Uploaded file must have a filename.")
        if uploaded_file is None and not query_file:
            raise ValueError("Provide query_file or upload an image file.")
        if query_file is not None and not isinstance(query_file, str):
            raise ValueError("query_file must be a string.")

        k = _parse_int(payload.get("k", DEFAULT_TOP_K), "k", min_value=1, max_value=MAX_TOP_K)
        limit = _parse_limit(payload.get("limit", DEFAULT_LIMIT))
        refresh_cache = _parse_bool(payload.get("refresh_cache", False), "refresh_cache")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    started_at = time.perf_counter()
    try:
        query_name, results = retrieve_matches(
            query_file=query_file,
            uploaded_file=uploaded_file,
            k=k,
            limit=limit,
            refresh_cache=refresh_cache,
        )
    except (TypeError, ValueError) as exc:
        _record_query_result(success=False)
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        _record_query_result(success=False)
        return jsonify({"error": str(exc)}), 404
    except RuntimeError as exc:
        _record_query_result(success=False)
        logger.warning("Retrieval runtime error: %s", exc)
        return jsonify({"error": str(exc)}), 500
    except Exception:
        _record_query_result(success=False)
        logger.exception("Unexpected retrieval failure")
        return jsonify({"error": "Internal server error during retrieval."}), 500

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    _record_query_result(success=True, latency_ms=latency_ms)

    return jsonify({
        "query": query_name,
        "count": len(results),
        "results": results,
        "database_connected": True,
        "latency_ms": latency_ms,
    })


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)
