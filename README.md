# SAR to Optical Image Retrieval

This project trains a simple deep learning retrieval system that matches SAR images with their corresponding optical satellite images. It uses two ResNet18-based encoders, one for SAR images and one for optical images, and trains them with a contrastive loss so matching image pairs have similar embeddings.

## Project Structure

```text
.
├── data/
│   └── Paired_SAR_Optical_images/   # paired SAR/optical image files
├── saved_model/
│   ├── sar_encoder.pth              # trained SAR encoder weights
│   ├── optical_encoder.pth          # trained optical encoder weights
│   ├── optical_embeddings.pt        # cached optical embeddings
│   └── optical_files.txt            # file order for cached embeddings
├── train.py                         # trains SAR and optical encoders
├── test_retrieval.py                # runs SAR-to-optical retrieval
└── rquirements.txt                  # dependency file, currently empty
```

The code supports either of these dataset layouts:

```text
data/
├── SAR/
└── Optical/
```

or:

```text
data/
└── Paired_SAR_Optical_images/
```

When `data/SAR` and `data/Optical` are not found, the scripts automatically use `data/Paired_SAR_Optical_images` for both SAR and optical inputs.

## Requirements

Install the main Python dependencies:

```bash
pip install torch torchvision pillow tqdm matplotlib
```

Recommended Python version: Python 3.10 or newer.

If you want to use the existing dependency file, note that it is named `rquirements.txt` instead of the usual `requirements.txt`, and it is currently empty.

## Training

Run:

```bash
python train.py
```

Training behavior:

- Uses CUDA automatically if available.
- Uses CPU fallback settings when CUDA is not available.
- Saves model checkpoints into `saved_model/`.
- Saves `sar_encoder.pth` and `optical_encoder.pth` after each epoch.

Default settings in `train.py`:

- GPU: 10 epochs, batch size 64.
- CPU: 3 epochs, batch size 32, first 2000 paired images only.
- Image size: 224 x 224.
- Embedding size: 256.

## Retrieval

After training, run retrieval with:

```bash
python test_retrieval.py
```

This uses the first available SAR image as the query and shows the top optical matches.

To use a specific query image:

```bash
python test_retrieval.py --query ROIs1868_summer_s1_59_p10.png
```

Useful options:

```bash
python test_retrieval.py --query <sar_filename> --k 5
python test_retrieval.py --limit 1000
python test_retrieval.py --refresh-cache
```

Options:

- `--query`: SAR image filename to search with.
- `--k`: number of top optical matches to display.
- `--limit`: limits the number of optical images encoded for faster testing.
- `--refresh-cache`: recomputes optical embeddings instead of using the saved cache.

## Saved Files

The retrieval script creates and reuses:

- `saved_model/optical_embeddings.pt`
- `saved_model/optical_files.txt`

These files cache optical image embeddings so retrieval is faster after the first run. Use `--refresh-cache` if the dataset changes.

## Notes

- The project currently has no separate evaluation metrics script.
- `matplotlib` is used to display the query image and top retrieval results.
- If a query filename is not found, `test_retrieval.py` prints a warning and falls back to the first available SAR file.
