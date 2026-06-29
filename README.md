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
├── data/
│   ├── SAR/                          # SAR images (alternative layout)
│   ├── Optical/                      # Optical images (alternative layout)
│   └── Paired_SAR_Optical_images/    # Paired images (single-directory layout)
├── saved_model/
│   ├── sar_encoder.pth               # Trained SAR encoder weights
│   ├── optical_encoder.pth           # Trained optical encoder weights
│   ├── optical_embeddings.pt         # Cached optical embedding vectors
│   └── optical_files.txt             # Filename index for the embedding cache
├── train.py                          # Model training script
├── test_retrieval.py                 # CLI retrieval evaluation script
├── flask_backend.py                  # REST API server
├── index.html                        # Web frontend
├── script.js                         # Frontend Javascript
├── styles.css                        # Frontend styles
└── rquirements.txt                   # Dependency list
```

---

## Requirements

**Python 3.10 or newer** is recommended.

Install dependencies:

```bash
pip install torch torchvision pillow tqdm matplotlib flask
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
python train.py
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
python test_retrieval.py

# Specify a query image
python test_retrieval.py --query ROIs1868_summer_s1_59_p10.png

# Return top-10 matches, limit the optical index to 1,000 images
python test_retrieval.py --query <filename> --k 10 --limit 1000

# Recompute optical embeddings (e.g. after dataset changes)
python test_retrieval.py --refresh-cache
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
python flask_backend.py
```

The API runs at `http://127.0.0.1:5000` by default.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check and device info |
| `GET` | `/api/status` | Dataset, model, and cache status |
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
  "database_connected": false
}
```

---

## Web Frontend

Open `index.html` in a browser (with the Flask API running) to use the GeoSat interface, which provides:

- **Search** — text query, image upload, and map-region query tabs with sensor type, date range, and cloud cover filters
- **Results** — image grid with cosine similarity scores and metadata tags
- **Image Viewer** — full-resolution view with sensor, date, coordinate, and resolution metadata
- **Dashboard** — query log with latency tracking

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

## Known Limitations

- No formal evaluation metrics script (Recall@K, mAP) is included.
- The web frontend currently uses mock data for the dashboard; live API wiring is partial.
- `rquirements.txt` is provided but currently empty; install dependencies manually as described above.
- The Flask server is intended for development use; deploy behind a production WSGI server (e.g. Gunicorn) for production workloads.

---

## License

This project is released under the [MIT License](LICENSE).
