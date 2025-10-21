# load data : done
# build embedding model : not done
# train embedding model : not done
# generate song embeddings : not done
# build the users prefence vectors : not done
# compare similarity between liked songs and canidate songs : not done
# display resuts/ recommend new songs : not done
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

canidate_tracks = pd.read_csv("candidate_tracks.csv")
liked_songs = pd.read_csv("liked_songs.csv")

features_columns = ['danceability', 'energy', 'valence', 'speechiness']
canidate_features = torch.tensor(canidate_tracks[features_columns].values,
                                 dtype=torch.float32)
liked_features = torch.tensor(liked_songs[features_columns].values,
                              dtype=torch.float32)

print("Canidate Features Tensor Shape:", canidate_features.shape)
print("Liked Features Tensor Shape:", liked_features.shape)
