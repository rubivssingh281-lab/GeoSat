import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

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

SAVE_DIR = 'saved_model'
os.makedirs(SAVE_DIR, exist_ok=True)

USE_CPU_FALLBACK = DEVICE.type == 'cpu'
DEFAULT_EPOCHS = 10 if not USE_CPU_FALLBACK else 3
BATCH_SIZE = 64 if not USE_CPU_FALLBACK else 32
MAX_TRAIN_SAMPLES = None if not USE_CPU_FALLBACK else 2000
NUM_WORKERS = 4 if not USE_CPU_FALLBACK else 0
PIN_MEMORY = not USE_CPU_FALLBACK


class SAROpticalDataset(Dataset):
    def __init__(self, sar_dir, optical_dir):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir

        sar_files = sorted(
            [f for f in os.listdir(sar_dir) if os.path.isfile(os.path.join(sar_dir, f))]
        )
        optical_files = sorted(
            [f for f in os.listdir(optical_dir) if os.path.isfile(os.path.join(optical_dir, f))]
        )

        self.files = sorted(list(set(sar_files).intersection(optical_files)))

        if len(self.files) == 0:
            raise RuntimeError(
                f'No paired image files found in {sar_dir} and {optical_dir}.'
            )

        self.transform_sar = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.transform_optical = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        sar_path = os.path.join(self.sar_dir, filename)
        optical_path = os.path.join(self.optical_dir, filename)

        sar_img = Image.open(sar_path).convert('L')
        optical_img = Image.open(optical_path).convert('RGB')

        sar_img = self.transform_sar(sar_img)
        optical_img = self.transform_optical(optical_img)

        return sar_img, optical_img, filename


class Encoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        try:
            base = models.resnet18(weights=None)
        except TypeError:
            base = models.resnet18(pretrained=False)
        in_features = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.projector(x)
        x = F.normalize(x, dim=1)
        return x


def contrastive_loss(sar_embed, optical_embed, temperature=0.07):
    logits = torch.matmul(sar_embed, optical_embed.T) / temperature
    labels = torch.arange(logits.size(0), device=DEVICE)
    loss_sar_to_optical = F.cross_entropy(logits, labels)
    loss_optical_to_sar = F.cross_entropy(logits.T, labels)
    return (loss_sar_to_optical + loss_optical_to_sar) / 2


def train():
    dataset = SAROpticalDataset(SAR_DIR, OPTICAL_DIR)

    if MAX_TRAIN_SAMPLES is not None and len(dataset) > MAX_TRAIN_SAMPLES:
        dataset.files = dataset.files[:MAX_TRAIN_SAMPLES]
        print(f'CPU fallback enabled: using first {MAX_TRAIN_SAMPLES} paired images')

    print(f'Using device: {DEVICE}')
    print(f'Dataset size: {len(dataset)} paired images')
    print(f'Epochs: {DEFAULT_EPOCHS}, Batch size: {BATCH_SIZE}')

    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    sar_encoder = Encoder().to(DEVICE)
    optical_encoder = Encoder().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(sar_encoder.parameters()) + list(optical_encoder.parameters()),
        lr=1e-4
    )

    EPOCHS = DEFAULT_EPOCHS

    for epoch in range(EPOCHS):
        sar_encoder.train()
        optical_encoder.train()
        total_loss = 0.0

        for sar_imgs, optical_imgs, _ in tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            sar_imgs = sar_imgs.to(DEVICE)
            optical_imgs = optical_imgs.to(DEVICE)

            sar_embed = sar_encoder(sar_imgs)
            optical_embed = optical_encoder(optical_imgs)

            loss = contrastive_loss(sar_embed, optical_embed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}')

        torch.save(sar_encoder.state_dict(), os.path.join(SAVE_DIR, 'sar_encoder.pth'))
        torch.save(optical_encoder.state_dict(), os.path.join(SAVE_DIR, 'optical_encoder.pth'))

    print('Training completed.')


if __name__ == '__main__':
    train()
