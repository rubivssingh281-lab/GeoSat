import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from train import Encoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = 'data'
SAR_DIR = os.path.join(BASE_DIR, 'SAR')
OPTICAL_DIR = os.path.join(BASE_DIR, 'Optical')
PAIRED_DIR = os.path.join(BASE_DIR, 'Paired_SAR_Optical_images')

if os.path.isdir(SAR_DIR) and os.path.isdir(OPTICAL_DIR):
    pass
elif os.path.isdir(PAIRED_DIR):
    SAR_DIR = PAIRED_DIR
    OPTICAL_DIR = PAIRED_DIR
else:
    raise FileNotFoundError(
        'Dataset not found. Create data/SAR and data/Optical directories, or use data/Paired_SAR_Optical_images.'
    )

SAR_MODEL = 'saved_model/sar_encoder.pth'
OPTICAL_MODEL = 'saved_model/optical_encoder.pth'
EMBEDDING_CACHE_PATH = os.path.join('saved_model', 'optical_embeddings.pt')
EMBEDDING_FILES_LIST = os.path.join('saved_model', 'optical_files.txt')

USE_CPU_FALLBACK = DEVICE.type == 'cpu'
DEFAULT_LIMIT = 2000 if USE_CPU_FALLBACK else None
BATCH_SIZE = 64 if not USE_CPU_FALLBACK else 32
NUM_WORKERS = 4 if not USE_CPU_FALLBACK else 0
PIN_MEMORY = not USE_CPU_FALLBACK

transform_sar = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_optical = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def load_image(path, transform, mode):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Image not found: {path}')
    img = Image.open(path).convert(mode)
    return transform(img).unsqueeze(0).to(DEVICE)


class ImagePathsDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, root_dir, transform, mode):
        self.file_paths = file_paths
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filename = self.file_paths[idx]
        path = os.path.join(self.root_dir, filename)
        img = Image.open(path).convert(self.mode)
        return self.transform(img), filename


def cache_is_valid(files):
    if not os.path.exists(EMBEDDING_CACHE_PATH) or not os.path.exists(EMBEDDING_FILES_LIST):
        return False
    with open(EMBEDDING_FILES_LIST, 'r', encoding='utf-8') as f:
        cached_files = [line.strip() for line in f.readlines()]
    return cached_files == files


def build_optical_database(optical_encoder, limit=None, refresh_cache=False):
    optical_encoder.eval()

    files = sorted(
        [f for f in os.listdir(OPTICAL_DIR) if os.path.isfile(os.path.join(OPTICAL_DIR, f))]
    )
    if len(files) == 0:
        raise RuntimeError(f'No optical files found in {OPTICAL_DIR}')
    if limit is not None:
        files = files[:limit]

    if not refresh_cache and cache_is_valid(files):
        print('Loading cached optical embeddings...')
        embeddings = torch.load(EMBEDDING_CACHE_PATH, map_location='cpu')
        return files, embeddings

    print(f'Computing optical embeddings for {len(files)} images...')
    dataset = ImagePathsDataset(files, OPTICAL_DIR, transform_optical, 'RGB')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    embeddings = []
    with torch.no_grad():
        for batch_images, _ in loader:
            batch_images = batch_images.to(DEVICE)
            batch_embeddings = optical_encoder(batch_images).cpu()
            embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0)
    torch.save(embeddings, EMBEDDING_CACHE_PATH)
    with open(EMBEDDING_FILES_LIST, 'w', encoding='utf-8') as f:
        for filename in files:
            f.write(filename + '\n')
    return files, embeddings


def retrieve_top_k(query_sar_file, k=5, limit=None, refresh_cache=False):
    if not os.path.exists(SAR_MODEL) or not os.path.exists(OPTICAL_MODEL):
        raise FileNotFoundError(
            'Model checkpoint not found. Run train.py first and confirm saved_model/*.pth exists.'
        )

    sar_encoder = Encoder().to(DEVICE)
    optical_encoder = Encoder().to(DEVICE)

    sar_encoder.load_state_dict(torch.load(SAR_MODEL, map_location=DEVICE))
    optical_encoder.load_state_dict(torch.load(OPTICAL_MODEL, map_location=DEVICE))

    sar_encoder.eval()
    optical_encoder.eval()

    optical_files, optical_embeddings = build_optical_database(
        optical_encoder, limit=limit, refresh_cache=refresh_cache
    )
    optical_embeddings = optical_embeddings.to(DEVICE)

    query_path = os.path.join(SAR_DIR, query_sar_file)
    query_img = load_image(query_path, transform_sar, 'L')

    with torch.no_grad():
        query_embedding = sar_encoder(query_img)

    similarities = torch.matmul(query_embedding, optical_embeddings.T).squeeze(0)
    top_scores, top_indices = torch.topk(similarities, k)

    print(f'Using device: {DEVICE}')
    print(f'Query SAR: {query_sar_file}')
    print('Top matches:')
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), start=1):
        print(f'{rank}. {optical_files[idx]} | Score: {score.item():.4f}')

    show_results(query_sar_file, optical_files, top_indices)


def show_results(query_sar_file, optical_files, top_indices):
    query_path = os.path.join(SAR_DIR, query_sar_file)
    query_img = Image.open(query_path).convert('L')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(top_indices) + 1, 1)
    plt.imshow(query_img, cmap='gray')
    plt.title('Query SAR')
    plt.axis('off')

    for i, idx in enumerate(top_indices):
        optical_path = os.path.join(OPTICAL_DIR, optical_files[idx])
        optical_img = Image.open(optical_path).convert('RGB')

        plt.subplot(1, len(top_indices) + 1, i + 2)
        plt.imshow(optical_img)
        plt.title(f'Top {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def get_default_query():
    files = sorted(
        [f for f in os.listdir(SAR_DIR) if os.path.isfile(os.path.join(SAR_DIR, f))]
    )
    if len(files) == 0:
        raise RuntimeError(f'No SAR files found in {SAR_DIR}')
    return files[0]


def validate_query_file(query_file):
    if query_file is None:
        return get_default_query()

    query_path = os.path.join(SAR_DIR, query_file)
    if os.path.isfile(query_path):
        return query_file

    files = sorted(
        [f for f in os.listdir(SAR_DIR) if os.path.isfile(os.path.join(SAR_DIR, f))]
    )
    if len(files) == 0:
        raise RuntimeError(f'No SAR files found in {SAR_DIR}')

    sample_files = ', '.join(files[:5])
    print(f"Warning: query file '{query_file}' not found in {SAR_DIR}.")
    print(f"Using default query '{files[0]}' instead. Sample available files: {sample_files}")
    return files[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test retrieval with trained encoders.')
    parser.add_argument('--query', type=str, default=None, help='SAR filename to use as query')
    parser.add_argument('--k', type=int, default=5, help='Number of optical matches to show')
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, help='Limit optical images to encode for faster retrieval')
    parser.add_argument('--refresh-cache', action='store_true', help='Force recomputing optical embeddings')
    args = parser.parse_args()

    query_file = validate_query_file(args.query)
    print(f'Query file: {query_file}')
    retrieve_top_k(query_file, k=args.k, limit=args.limit, refresh_cache=args.refresh_cache)
