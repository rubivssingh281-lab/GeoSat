# GeoSat — SAR-to-Optical Satellite Image Retrieval

A deep learning system that matches **Synthetic Aperture Radar (SAR)** images with their corresponding **optical satellite** counterparts using learned cross-modal embeddings. Includes a Flask REST API and a browser-based frontend for querying the image archive.

---

## Overview

GeoSat trains two ResNet18-based encoder networks — one for SAR imagery and one for optical imagery — using contrastive learning. At inference time, a SAR query image is encoded into an embedding vector and matched against a pre-computed optical image database via cosine similarity, returning the top-*k* most similar optical scenes.

**Key features**

- Dual-encoder architecture with a shared contrastive loss (InfoNCE)
- Embedding cache for fast repeated retrieval without re-encoding
- Flask API with endpoints for retrieval, image serving, and system status
- Web UI (GeoSat) with search, results grid, image viewer, and query dashboard
- Automatic CPU fallback with reduced batch size and epoch count
- Supports both paired and separate SAR/Optical dataset layouts

---

## Project Structure

```
.
├── backend/
│   ├── data/                         # SAR & Optical satellite images
│   │   ├── SAR/
│   │   ├── Optical/
│   │   └── Paired_SAR_Optical_images/
│   ├── saved_model/                  # Encoders & 2000-image embedding cache
│   │   ├── sar_encoder.pth
│   │   ├── optical_encoder.pth
│   │   ├── optical_embeddings.pt
│   │   ├── optical_files.txt
│   │   └── optical_cache.json
│   ├── flask_backend.py              # REST API server (port 5000)
│   ├── train.py                      # Dual-encoder contrastive training
│   ├── test_retrieval.py             # CLI retrieval evaluation script
│   ├── test_security.py              # Automated cybersecurity test suite
│   └── requirements.txt              # Backend dependencies
├── frontend/
│   ├── index.html                    # Decoupled web interface
│   ├── css/
│   │   └── styles.css                # Full UI stylesheet
│   ├── js/
│   │   └── script.js                 # Frontend API client & dashboard
│   └── images/                       # UI assets and brand icons
├── run.bat                           # 1-click launcher for Windows
├── run.ps1                           # PowerShell launcher
└── requirements.txt                  # Project dependencies
```

---

## Requirements

**Python 3.10 or newer** is recommended.

Install dependencies in a virtual environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

> **GPU:** CUDA is used automatically when available. CPU execution is supported but significantly slower.

---

## Dataset Layout

The project supports two dataset directory structures. Place your data under `data/` using either layout:

**Layout A — Separate directories**
```
data/
├── SAR/
│   ├── image_001.png
│   └── ...
└── Optical/
    ├── image_001.png
    └── ...
```

**Layout B — Single paired directory**
```
data/
└── Paired_SAR_Optical_images/
    ├── image_001.png
    └── ...
```

When Layout A directories are not found, the scripts automatically fall back to Layout B, using the same directory for both SAR and optical inputs.

> Paired images must share the same filename across SAR and Optical directories.

---

## Training

```bash
python backend/train.py
```

The script trains both encoders end-to-end using a symmetric InfoNCE contrastive loss. Checkpoints are saved to `saved_model/` after each epoch.

**Default training configuration**

| Setting | GPU | CPU fallback |
|---|---|---|
| Epochs | 10 | 3 |
| Batch size | 64 | 32 |
| Max training samples | All | 2,000 |
| Image size | 224 × 224 | 224 × 224 |
| Embedding dimension | 256 | 256 |
| Workers | 4 | 0 |

---

## Retrieval (CLI)

After training, run retrieval from the command line:

```bash
# Use the first available SAR image as the query
python backend/test_retrieval.py

# Specify a query image
python backend/test_retrieval.py --query ROIs1868_summer_s1_59_p10.png

# Return top-10 matches, limit the optical index to 1,000 images
python backend/test_retrieval.py --query <filename> --k 10 --limit 1000

# Recompute optical embeddings (e.g. after dataset changes)
python backend/test_retrieval.py --refresh-cache
```

**Options**

| Flag | Default | Description |
|---|---|---|
| `--query` | First SAR file | SAR image filename to use as the query |
| `--k` | `5` | Number of top optical matches to return |
| `--limit` | `2000` (CPU) / all (GPU) | Maximum number of optical images to index |
| `--refresh-cache` | `false` | Force recomputation of optical embeddings |

Results are displayed with match scores and visualised using `matplotlib`.

---

## REST API

Start the Flask development server:

```bash
python backend/flask_backend.py
```

The API runs at `http://127.0.0.1:5000` by default.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check and device info |
| `GET` | `/api/status` | Dataset, model, and cache status |
| `GET` | `/api/metrics` | In-process query counts and average latency |
| `GET` | `/api/images/<kind>` | List images (`sar` or `optical`) |
| `GET` | `/api/image/<kind>/<filename>` | Serve an individual image |
| `POST` | `/api/retrieve` | Run SAR-to-optical retrieval |

### Retrieval request

**JSON body (filename query)**
```json
{
  "query_file": "ROIs1868_summer_s1_59_p10.png",
  "k": 6,
  "limit": 2000,
  "refresh_cache": false
}
```

**Multipart form (file upload)**
```
POST /api/retrieve
Content-Type: multipart/form-data

file=<sar_image_file>
k=6
```

**Response**
```json
{
  "query": "ROIs1868_summer_s1_59_p10.png",
  "count": 6,
  "results": [
    {
      "rank": 1,
      "filename": "ROIs1868_summer_s2_59_p10.png",
      "score": 0.941872,
      "match_percent": 94,
      "image_url": "/api/image/optical/ROIs1868_summer_s2_59_p10.png"
    }
  ],
  "database_connected": true,
  "latency_ms": 184.52
}
```

The backend resolves data and model paths relative to `flask_backend.py`, so it
can be started from any working directory. Invalid `k`, `limit`, missing query
files, and unsafe image paths return an appropriate `4xx` response.

`/api/metrics` reports successful and failed retrieval counts plus the average
latency for the most recent 50 successful queries. Metrics are intentionally
in-memory and reset when the development server restarts.

---

## Web Frontend

Start the API, then serve the frontend over HTTP (opening the file directly can
block browser requests):

```bash
.venv/bin/python flask_backend.py
# In a second terminal (serve frontend directory):
python -m http.server 8080 -d frontend
```

Or simply launch everything with one click on Windows:
```cmd
run.bat
```

Open `http://127.0.0.1:8080` to use the GeoSat interface, which provides:

- **Search** — text query, image upload, and map-region query tabs with sensor type, date range, and cloud cover filters
- **Results** — image grid with cosine similarity scores and metadata tags
- **Image Viewer** — full-resolution view with sensor, date, coordinate, and resolution metadata
- **Dashboard** — live backend/device/cache status and query log with latency tracking

---

## Embedding Cache

On first retrieval, optical embeddings are computed and saved to:

```
saved_model/optical_embeddings.pt   # Embedding tensor
saved_model/optical_files.txt       # Corresponding filename index
```

Subsequent runs load from cache automatically, provided the file list has not changed. Use `--refresh-cache` (CLI) or `"refresh_cache": true` (API) to invalidate and rebuild.

---

## Architecture

```
SAR image  ──►  SAR Encoder  (ResNet18 + projection head)  ──►  256-d L2-normalised embedding
                                                                         │
                                                                  cosine similarity
                                                                         │
Optical DB ──►  Optical Encoder (ResNet18 + projection head) ──►  256-d L2-normalised embeddings
```

Both encoders share the same architecture: a ResNet18 backbone with the classification head replaced by a two-layer MLP projection head (`512 → ReLU → 256`), followed by L2 normalisation. The contrastive loss is computed symmetrically across SAR-to-optical and optical-to-SAR directions.

---

## Cybersecurity & Hardening Architecture

GeoSat implements enterprise-grade application security defenses to protect against OWASP Top 10 risks, model poisoning, and Denial of Service (DoS):

### 1. Insecure Deserialization Defense
- PyTorch checkpoint loading via `safe_torch_load` strictly enforces `weights_only=True` to prevent arbitrary Python bytecode execution (RCE) via malicious pickle payloads in model weights or embedding caches.

### 2. File Upload Hardening & Pixel Flood Protection
- **Magic-Byte Signature Verification**: Uploaded images are inspected for binary magic bytes (`PNG`, `JPEG`, `BMP`, `TIFF`, `WebP`) prior to PIL processing, preventing polyglot file uploads and script execution.
- **Decompression Bomb Defense**: PIL `Image.MAX_IMAGE_PIXELS` is capped at 25,000,000 pixels with maximum payload upload body bounded to 25 MB (`GEOSAT_MAX_UPLOAD_MB`).

### 3. Path Traversal & Null-Byte Neutralization
- `resolve_safe_image_path` sanitizes image requests, rejects null bytes (`\x00`), and verifies real paths with `os.path.commonpath` to eliminate directory traversal (`../../`), junction escapes, and LFI vulnerabilities.

### 4. Sliding Window Rate Limiting & Resource Exhaustion Defense
- In-memory thread-safe `SlidingWindowRateLimiter` enforces rate limiting on `/api/retrieve` (configurable via `GEOSAT_RATE_LIMIT_PER_MINUTE`, default 60 req/min). Exceeded rates respond with HTTP `429 Too Many Requests` and standard `Retry-After` headers.

### 5. Strict HTTP Security Headers & Origin Validation
Every response includes hardened HTTP defense headers:
- `X-Content-Type-Options: nosniff` (MIME confusion mitigation)
- `X-Frame-Options: DENY` (Clickjacking defense)
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy` (Restricting scripts, styles, fonts, and connect origins)
- `Permissions-Policy` (Disabling camera, microphone, geolocation, and USB APIs)
- `X-Request-ID` (UUID correlation tracing across queries)
- Configurable CORS origin validation (`GEOSAT_CORS_ORIGINS`).

### 6. RFC 9457 Problem Details Contract
- All HTTP errors conform to RFC 9457 (`application/problem+json`) with standard `type`, `title`, `status`, `detail`, `instance`, and `request_id` keys, retaining backward compatibility for client consumers.

### 7. Frontend DOM XSS Protection
- The frontend strictly sanitizes all search result image URLs, allowing only internal relative paths and explicit `http:`/`https:` web protocols, neutralizing `javascript:` or `data:` protocol injection.

---

## Known Limitations

- No formal evaluation metrics script (Recall@K, mAP) is included.
- Text and map-region search tabs are UI placeholders; image upload is the supported query mode.
- Query metrics are process-local and are not intended to replace production monitoring.
- The Flask server is intended for development use; deploy behind a production WSGI server (e.g. Gunicorn) for production workloads.

---

## License

This project is released under the [MIT License](LICENSE).
