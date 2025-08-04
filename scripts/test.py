import torch
import torchaudio
import numpy as np
from pathlib import Path
from model import CNNGenreClassifier
from data import DatasetPrep
import librosa
import librosa.display

SR = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
SEGMENT_DURATION = 10.0

DEVICE = torch.device("cuda")

genres = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]
model = CNNGenreClassifier(numClasses=len(genres)).to(DEVICE)
model.load_state_dict(torch.load("models/best_model.pt", map_location=DEVICE))
model.eval()

def extract_mels(filePath):
    y, sr = librosa.load(filePath, sr=SR, mono=True)
    samplesPerSegment = int(SEGMENT_DURATION * SR)
    numSegments = len(y) // samplesPerSegment
    segments = []

    for s in range(numSegments):
        start = s * samplesPerSegment
        end = start + samplesPerSegment
        segment = y[start:end]
        mel = librosa.feature.melspectrogram(
            y=segment, sr=SR, n_mels=N_MELS,
            hop_length=HOP_LENGTH, n_fft=N_FFT)
        melDb = librosa.power_to_db(mel, ref=np.max)
        segments.append(melDb)
    return segments

def predict(filePath):
    segments = extract_mels(filePath)
    predictions = []
    with torch.no_grad():
        for mel in segments:
            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            out = model(x)
            predictions.append(out.cpu())
    avgLogits = torch.mean(torch.cat(predictions), dim=0)

    allPreds = [torch.argmax(p, dim=1) for p in predictions]
    predId = torch.mode(torch.cat(allPreds)).values.item()
    return genres[predId]

if __name__ == "__main__":
    testFile = "data/metal_music.mp3"
    print("Predicted genre:", predict(testFile))