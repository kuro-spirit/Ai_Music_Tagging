import sys
sys.path.append('../')

import torch
from torch import nn, optim
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from data import DatasetPrep
from model import CNNGenreClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 10
PROCESSED_PATH = "data/processed"
LR = 1e-3

def evaluate(model, loader, criterion):
    model.eval()
    totalLoss = 0
    correct = 0
    allPreds = []
    allLabels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            totalLoss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            allPreds.extend(preds.cpu().numpy())
            allLabels.extend(y.cpu().numpy())

    avgLoss = totalLoss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avgLoss, accuracy, np.array(allLabels), np.array(allPreds)


def main():
    trainDataset = DatasetPrep(PROCESSED_PATH, split='train', preload=True)
    testDataset = DatasetPrep(PROCESSED_PATH, split='test', preload=True)
    numClasses = len(trainDataset.genres)

    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    testLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, num_workers=12, pin_memory=True)

    # Inspect shape
    exampleBatch, _ = next(iter(trainLoader))
    print("Example batch shape:", exampleBatch.shape)

    model = CNNGenreClassifier(numClasses).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Model running on:", next(model.parameters()).device)
    print("Train samples:", len(trainDataset))
    print("Validation samples:", len(testDataset))
    
    bestTestAcc = 0

    trainLosses, testLosses = [], []
    trainAccs, testAccs = [], []

    scaler = GradScaler()

    startTime = dt.now()

    for epoch in range(EPOCHS):
        model.train()
        totalLoss = 0
        correct = 0
        for x, y in trainLoader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = model(x)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            totalLoss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()

        trainAcc = correct / len(trainDataset)
        avgTrainLoss = totalLoss / len(trainLoader)
        
        testLoss, testAcc, testLabels, testPreds = evaluate(model, testLoader, criterion)

        print(f"Epoch {epoch+1}: "
              f"trainLoss={avgTrainLoss:.4f}, trainAcc={trainAcc:.3f}, "
              f"testLoss={testLoss:.4f}, testAcc={testAcc:.3f}")
        
        if testAcc > bestTestAcc:
            bestTestAcc = testAcc
            torch.save(model.state_dict(), "best_model.pt")

        # after each epoch
        trainLosses.append(avgTrainLoss)
        trainAccs.append(trainAcc)
        testLosses.append(testLoss)
        testAccs.append(testAcc)


    endTime = dt.now()

    cm = confusion_matrix(testLabels, testPreds)
    print("Confusion matrix:\n", cm)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(trainLosses, label='Train')
    plt.plot(testLosses, label='Test')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(trainAccs, label='Train')
    plt.plot(testAccs, label='Test')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    print(f"Run time: {endTime - startTime}")

if __name__ == "__main__":
    main()
