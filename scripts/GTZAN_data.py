from pathlib import Path
import numpy as np
import torch
import torchaudio
import random
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class DatasetPrep(Dataset):
    def __init__(self, processedDir, split='train', testSize=0.2, randomState=42, preload=True):
        self.processedDir = Path(processedDir)
        self.genres = sorted([p.name for p in self.processedDir.iterdir() if p.is_dir()])

        # collect files
        allFiles = []
        allLabels = []
        for idx, genre in enumerate(self.genres):
            files = list((self.processedDir / genre).glob("*.npy"))
            allFiles.extend(files)
            allLabels.extend([idx] * len(files))

        # Train/Test set split
        trainFiles, testFiles, trainLabels, testLabels= train_test_split(
            allFiles, allLabels, test_size=testSize, stratify=allLabels, random_state=randomState
            )
        
        if split=='train':
            self.files = trainFiles
            self.labels = trainLabels
            self.specAugment = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=30)
            )
        elif split == 'test':
            self.specAugment = None
            self.files = testFiles
            self.labels = testLabels

        self.preload = preload
        if self.preload:
            print(f"Preloading {len(self.files)} spectrograms into RAM...")
            self.data = [np.load(f) for f in self.files]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # start = time.time()
        if self.preload:
            mel = self.data[idx]
        else:
            mel = np.load(self.files[idx])
        # print("getitem time:", time.time()-start)

        melTensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        if self.specAugment and random.random() < 0.5:
            melTensor = self.specAugment(melTensor)

        labelTensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return melTensor, labelTensor
