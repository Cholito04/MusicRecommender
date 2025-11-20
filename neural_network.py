import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load track data and pick the features we care about
candidate_df = pd.read_csv("candidate_tracks.csv")
liked_songs = pd.read_csv("liked_songs.csv")
features_columns = ["danceability", "energy", "valence", "speechiness",
                    "acousticness", "instrumentalness",
                    "liveness", "tempo", "loudness"]

# convert features to tensors
X_liked = torch.tensor(liked_songs[features_columns].values,
                       dtype=torch.float32).to(device)
X_candidate = torch.tensor(candidate_df[features_columns].values,
                           dtype=torch.float32).to(device)

print("Candidate Features Tensor Shape:", X_candidate.shape)
print("Liked Features Tensor Shape:", X_liked.shape)


# simple feedforward network to map song features into embedding space
class SimpleEmbeddingModel(nn.Module):
    def __init__(self, input_size=9, hidden_sizes=[32, 16], embedding_size=8):
        super().__init__()
        layers = [
            nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], embedding_size)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# train embeddings based on positive-only contrastive loss
def train_playlist_embeddings(model, X_liked, epochs=100, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # embed all liked songs
        emb = F.normalize(model(X_liked), dim=1)

        # compute all pairwise cosine similarities
        sim_matrix = torch.matmul(emb, emb.T)  # (N x N)

        # positive-only contrastive loss:
        # want all pairs to be similar → maximize similarity → minimize (1 - sim)
        # ignore diagonal (self similarity)
        N = sim_matrix.size(0)
        mask = ~torch.eye(N, dtype=bool, device=emb.device)
        pos_sims = sim_matrix[mask]

        loss = (1 - pos_sims).mean()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model


# generate recommendations by comparing candidate songs to playlist taste
def recommend_songs(model, X_liked, X_candidate, candidate_df, top_k=3):
    model.eval()
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

    return (playlist_emb.squeeze(0).cpu(), cand_emb.cpu(), top_idx,
            rec_songs, top_scores)


# main flow: train, then recommend
def main():
    model = SimpleEmbeddingModel().to(device)

    print("\n=== TRAINING EMBEDDINGS ===")
    model = train_playlist_embeddings(model, X_liked)

    print("\n=== GENERATING RECOMMENDATIONS ===")
    playlist_emb, cand_emb, top_idx, rec_songs, scores = recommend_songs(
        model, X_liked, X_candidate, candidate_df)

    # playlist embedding vector
    print("\n=== Playlist Embedding ===")
    print(playlist_emb.numpy())

    # show top recommendations with similarity
    print("\n=== Top Recommendations ===")
    for rank, i in enumerate(top_idx):
        song = candidate_df.iloc[i.item()]
        print(f"{rank+1}: {song['track_name']} ({song['artist_name']}), Score: {scores[rank].item():.4f}")
        print("Embedding:", cand_emb[i].numpy())
        print()

    print("\n=== Final Recommendations ===")
    print(rec_songs[["track_name", "artist_name"]])


if __name__ == "__main__":
    main()
