import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import sqlite3

from .embedding_model import SimpleEmbeddingModel


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features_columns = ["danceability", "energy", "valence", "speechiness",
                    "acousticness", "instrumentalness",
                    "liveness", "tempo", "loudness"]


# set up database
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, "../databases/my_database.db")


# load track data and pick the features we care about
def load_candidate_songs():
    conn = sqlite3.connect(db_path)

    query = "SELECT * FROM songs"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_playlist_songs(playlist_id):
    conn = sqlite3.connect(db_path)

    query = """
    SELECT * FROM songs

    JOIN playlist_songs ON songs.track_id = playlist_songs.track_id
    WHERE playlist_songs.playlist_id = ?
    """

    df = pd.read_sql_query(query, conn, params=(playlist_id,))
    conn.close()
    return df


def get_last_playlist_id():
    """Get the most recently added playlist"""
    conn = sqlite3.connect(db_path)
    query = (
        "SELECT playlist_id FROM playlists ORDER BY created_at DESC LIMIT 1")
    df = pd.read_sql_query(query, conn)
    conn.close()
    if len(df) == 0:
        raise ValueError("No playlists found in the database.")
    return df.iloc[0]['playlist_id']


def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "model_weights.pth")
    model = SimpleEmbeddingModel().to(device)
    model.load_state_dict(torch.load(weights_path,
                                     map_location=device, weights_only=True))
    model.eval()
    return model


# generate recommendations by comparing candidate songs to playlist taste
def recommend_songs(X_liked, X_candidate, candidate_df, top_k=3):
    model = load_model()

    with torch.no_grad():
        # get embeddings for liked songs and normalize
        liked_emb = F.normalize(model(X_liked), dim=1)

        # average embedding represents overall playlist taste
        playlist_emb = F.normalize(liked_emb.mean(dim=0, keepdim=True), dim=1)

        # embeddings for candidate songs
        cand_emb = F.normalize(model(X_candidate), dim=1)

        # cosine similarity between playlist and candidates
        scores = nn.CosineSimilarity(dim=1)(cand_emb,
                                            playlist_emb.expand_as(cand_emb))

        # pick top songs
        top_k = min(top_k, scores.size(0))
        top_scores, top_idx = torch.topk(scores, top_k)
        rec_songs = candidate_df.iloc[top_idx.cpu().numpy()].copy()

    return (playlist_emb.squeeze(0).cpu(), cand_emb.cpu(), liked_emb.cpu(),
            top_idx, rec_songs, top_scores)


# main flow: train, then recommend
def main():

    candidate_df = load_candidate_songs()

    # convert features to tensors
    X_candidate = torch.tensor(candidate_df[features_columns].values,
                               dtype=torch.float32).to(device)

    playlist_id = get_last_playlist_id()
    liked_songs = load_playlist_songs(playlist_id)
    X_liked = torch.tensor(liked_songs[features_columns].values,
                           dtype=torch.float32).to(device)

    print("\n=== GENERATING RECOMMENDATIONS ===")
    (playlist_emb, cand_emb, liked_emb, top_idx, rec_songs, scores) = (
        recommend_songs(X_liked, X_candidate, candidate_df))

    # playlist embedding vector
    print("\n=== Playlist Embedding ===")
    print(playlist_emb.numpy())

    # show top recommendations with similarity
    print("\n=== Top Recommendations ===")
    for rank, idx in enumerate(top_idx):
        song = candidate_df.iloc[idx.item()]
        # FIX 6: index cand_emb by rank (position in top-k),
        # not by idx (position in full candidate set)
        print(f"{rank+1}: {song['track_name']} ({song['artist']}), "
              f"Score: {scores[rank].item():.4f}")
        print("Embedding:", cand_emb[rank].numpy())
        print()

    print("\n=== Final Recommendations ===")
    print(rec_songs[["track_name", "artist"]])


if __name__ == "__main__":
    main()
