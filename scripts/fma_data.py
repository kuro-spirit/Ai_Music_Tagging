import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

EPS = 1e-8
MAX_FRAMES = 1280

class FMADataset(Dataset):
    def __init__(self, processed_path, split='train', preload=False):
        self.mel_dir = os.path.join(processed_path, "melSpec")
        self.labels_df = pd.read_csv(os.path.join(processed_path, "labels.csv"))

        # If you have split info (optional)
        split_csv = os.path.join(processed_path, f"{split}_split.csv")
        if os.path.exists(split_csv):
            split_ids = pd.read_csv(split_csv)['trackId'].astype(str).str.zfill(6)
            self.labels_df = self.labels_df[self.labels_df['trackId'].astype(str).str.zfill(6).isin(split_ids)]

        self.preload = preload
        if self.preload:
            self.data = []
            for idx, row in self.labels_df.iterrows():
                mel = np.load(os.path.join(self.mel_dir, f"{int(row['trackId']):06d}.npy"))
                mel = (mel - mel.min()) / (mel.max() - mel.min())
                self.data.append((mel, row['genreId']))

        self.genres = sorted(self.labels_df['genreId'].unique())

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if self.preload:
            mel, label = self.data[idx]
        else:
            row = self.labels_df.iloc[idx]
            track_id_str = f"{int(row['trackId']):06d}"
            folder = track_id_str[:3]  # e.g., 005156 -> '005'
            path = os.path.join(self.mel_dir, folder, f"{track_id_str}.npy")
            mel = np.load(path)
            mel = (mel - mel.min()) / (mel.max() - mel.min() + EPS)
            label = row['genreId']

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, time)
        label = torch.tensor(label, dtype=torch.long)
        return mel, label
