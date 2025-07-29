import sys
sys.path.append('../')

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from data import processed
from model import CNNGenreClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 10
PROCESSED_PATH = "data/processed"

def main():
    trainDataset = processed(PROCESSED_PATH, split='train')
    testDataset = processed(PROCESSED_PATH, split='val')
    numClasses = len(trainDataset.genres)

    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=BATCH_SIZE)

    # Inspect shape
    exampleBatch, _ = next(iter(trainLoader))
    print("Example batch shape:", exampleBatch.shape)

    model = CNNGenreClassifier(numClasses).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        totalLost = 0
        correct = 0
        for x, y in trainLoader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            totalLost += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()

        trainAcc = correct / len(trainDataset)
        print(f"Epoch {epoch+1}: loss={totalLost:.2f} acc={trainAcc:.3f}")

if __name__ == "__main__":
    main()
