import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNGenreClassifier(nn.Module):
    def __init__(self, numClasses, input_shape=(1, 128, 1291)):
        super().__init__()
        # Filters that slide over image, and each filter detects local patterns. 
        # Different layers find different patterns
        # Batch normalization to reduce overfitting by getting the average of each batch 
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, numClasses)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x