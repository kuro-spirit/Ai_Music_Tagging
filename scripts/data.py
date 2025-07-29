from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class DatasetPrep(Dataset):
    def __init__(self, processedDir, split='train', testSize=0.2, randomState=42):
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
        else:
            self.files = testFiles
            self.labels = testLabels

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        mel = np.load(self.files[idx])
        melTensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        labelTensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return melTensor, labelTensor
