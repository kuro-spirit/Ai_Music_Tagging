import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torchaudio.transforms as T

EPS = 1e-8
MAX_FRAMES = 1280

class FMADataset(Dataset):
    def __init__(self, processed_path, split='train', preload=False, transform=None, augment=False):
        self.mel_dir = os.path.join(processed_path, "melSpec")
        self.labels_df = pd.read_csv(os.path.join(processed_path, "labels.csv"))

        # Split filtering
        split_csv = os.path.join(processed_path, f"{split}_split.csv")
        if os.path.exists(split_csv):
            split_ids = pd.read_csv(split_csv)['trackId'].astype(str).str.zfill(6)
            self.labels_df = self.labels_df[self.labels_df['trackId'].astype(str).str.zfill(6).isin(split_ids)]

        # Genre mapping
        unique_genres = sorted(self.labels_df['genreId'].unique())
        self.genre_to_label = {genre: idx for idx, genre in enumerate(unique_genres)}
        self.labels_df['genreId'] = self.labels_df['genreId'].map(self.genre_to_label)

        self.preload = preload
        if self.preload:
            self.data = []
            for _, row in self.labels_df.iterrows():
                mel = np.load(os.path.join(self.mel_dir, f"{int(row['trackId']):06d}.npy"))
                mel = (mel - mel.min()) / (mel.max() - mel.min() + EPS)
                self.data.append((mel, row['genreId']))

        self.genres = list(self.genre_to_label.values())

        # Optional external transform (e.g., torchvision)
        self.transform = transform  

        # Augmentation flag — only for training
        self.augment = augment
        if self.augment:
            self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
            self.time_mask = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if self.preload:
            mel, label = self.data[idx]
        else:
            row = self.labels_df.iloc[idx]
            track_id_str = f"{int(row['trackId']):06d}"
            folder = track_id_str[:3]
            path = os.path.join(self.mel_dir, folder, f"{track_id_str}.npy")
            mel = np.load(path)
            mel = (mel - mel.min()) / (mel.max() - mel.min() + EPS)
            label = row['genreId']

        # Pad / crop
        if mel.shape[1] < MAX_FRAMES:
            pad_width = MAX_FRAMES - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel = mel[:, :MAX_FRAMES]

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, time)

        # Apply SpecAugment only if enabled
        if self.augment:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)

        # Apply any extra transforms (image-like)
        if self.transform:
            mel = self.transform(mel)

        label = torch.tensor(label, dtype=torch.long)
        return mel, label
