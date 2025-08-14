import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

dataset = "data/spotify.csv"

# Getting dataset into pd dataframe
df = pd.read_csv(dataset).iloc[:,1:]
df = df.drop_duplicates()

# Remove classifications where language is a genre
nonSoundGenre = ['british','french','german','swedish','spanish','indian','malay','turkish','world-music','gospel']
df = df.drop(df[df['track_genre'].isin(nonSoundGenre)].index)

# Drop non explanatory features
df = df.drop(columns = ['track_id','artists','album_name','track_name'])

# Map the explicit column to binary values
df['explicit'] = df['explicit'].map({False: 0,True: 1})

# Perform one-hot encoding
df = pd.get_dummies(df, columns=['key'], prefix = 'key', drop_first=True)
df = pd.get_dummies(df, columns=['time_signature'], prefix = 'meter', drop_first=True)

# Scaling function
def perform_scaling(df, colToScale, scalreType):
    # scalreType == 1 for StandardScaler, 2 for MinMaxScaler
    if scalreType == 1:
        scaler = StandardScaler()
    elif scalreType == 2:
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scalreType. Use 1 for StandardScaler or 2 for MinMaxScaler.")

    scaledData = scaler.fit_transform(df[colToScale])
    scaledDf = pd.DataFrame(scaledData, columns=colToScale)
    dfScaled = pd.concat([df.drop(columns=colToScale).reset_index(drop=True), scaledDf], axis=1)

    return dfScaled
  
# Define numeric columns
numericCol = ["popularity","duration_ms","danceability","energy","loudness","speechiness","acousticness","instrumentalness","liveness","valence","tempo"]
  
# Perform Scaling
df = perform_scaling(df, numericCol, 1)