import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import sqlite3
import os

# import nn model
from embedding_model import SimpleEmbeddingModel


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features_columns = ["danceability", "energy", "valence", "speechiness",
                    "acousticness", "instrumentalness",
                    "liveness", "tempo", "loudness"]


def load_candidate_songs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "../databases/my_database.db")
    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        danceability,
        energy,
        valence,
        speechiness,
        acousticness,
        instrumentalness,
        liveness,
        tempo,
        loudness
    FROM songs
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


# train embeddings based on positive-only contrastive loss
def train_playlist_embeddings(model, X_candidate, epochs=30, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # embed all liked songs
        emb = F.normalize(model(X_candidate), dim=1)

        # compute all pairwise cosine similarities
        sim_matrix = torch.matmul(emb, emb.T)  # (N x N)

        # positive-only contrastive loss:
        # want all pairs to be similar, maximize similarity, minimize (1 - sim)
        # ignore diagonal (self similarity)
        N = sim_matrix.size(0)
        mask = ~torch.eye(N, dtype=bool, device=emb.device)
        pos_sims = sim_matrix[mask]

        loss = (1 - pos_sims).mean()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), "model_weights.pth")
    return model


# main flow: train, then recommend
def main():

    # load track data and pick the features we care about
    candidate_df = load_candidate_songs()

    # convert features to tensors
    X_candidate = torch.tensor(candidate_df[features_columns].values,
                               dtype=torch.float32).to(device)

    model = SimpleEmbeddingModel().to(device)

    print("\n=== TRAINING EMBEDDINGS ===")
    model = train_playlist_embeddings(model, X_candidate)
    print("Training complete. Weights saved to model_weights.pth")


if __name__ == "__main__":
    main()
