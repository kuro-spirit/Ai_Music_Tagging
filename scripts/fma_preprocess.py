# fma_preprocess.py
import os
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

FMA_DIR = "data/fma_small"
TRACKS_CSV = "data/fma_metadata/tracks.csv"
OUTPUT_DIR = "data/fma_spectrograms"

def load_genre_mapping(metadataCsvPath):
    df = pd.read_csv(metadataCsvPath, index_col=0, header=[0, 1])
    genreLabels = df['track']['genre_top']
    validId = genreLabels[genreLabels.notnull()].index
    genreLabels = genreLabels.dropna()
    genreNames = genreLabels.unique()
    genreId = {genre: idx for idx, genre in enumerate(sorted(genreNames))}
    return genreLabels, genreId

def extract_mel_spectrogram(trackId, audioDir, sr=22050, n_mels=128, duration=30):
    sTrackId = f"{int(trackId):06d}"
    subdir = sTrackId[:3]
    filePath = os.path.join(audioDir, subdir, f"{sTrackId}.mp3")
    try:
        y, _ = librosa.load(filePath, sr=sr, duration=duration)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        melDb = librosa.power_to_db(mel, ref=np.max)
        return melDb
    except Exception as e:
        print(f"Error processing {filePath}: {e}")
        return None

def preprocess_all(fmaDir, metadataCsv, outputDir):
    os.makedirs(os.path.join(outputDir, "melSpec"), exist_ok=True)
    genreLabels, genreToId = load_genre_mapping(metadataCsv)

    labelData = []
    for trackId in tqdm(genreLabels.index):
        mel = extract_mel_spectrogram(trackId, fmaDir)
        if mel is not None:
            np.save(os.path.join(outputDir, "melSpec", f"{int(trackId):06d}.npy"), mel)
            genreName = genreLabels.loc[trackId]
            genreId = genreToId[genreName]
            labelData.append((int(trackId), genreId))

    df = pd.DataFrame(labelData, columns=["trackId", "genreId"])
    df.to_csv(os.path.join(outputDir, "labels.csv"), index=False)

    # Optionally save the mapping
    with open(os.path.join(outputDir, "genre_mapping.json"), "w") as f:
        json.dump(genreId, f)

if __name__ == "__main__":
    preprocess_all(FMA_DIR, TRACKS_CSV, OUTPUT_DIR)

    sample_id = "000005"
    path = f"data/fma_spectrograms/melSpec/{sample_id}.npy"

    mel = np.load(path)
    print("Mel Spectrogram shape:", mel.shape)

    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='magma')
    plt.title(f"Mel Spectrogram for Track {sample_id}")
    plt.xlabel("Time frames")
    plt.ylabel("Mel frequency bins")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()