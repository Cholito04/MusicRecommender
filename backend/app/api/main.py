
from fastapi import FastAPI, HTTPException
from ml.inference import (recommend_songs, load_candidate_songs,
                          load_playlist_songs)
from services.pull_data import (normalize_features, fetch_playlist_features,
                                save_to_db)
from .users import router as users_router
import torch
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Music Recommender")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Your React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include the users router
app.include_router(users_router)


class PlaylistData(BaseModel):
    username: str
    playlist_url: str


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_columns = ["danceability", "energy", "valence", "speechiness",
                    "acousticness", "instrumentalness",
                    "liveness", "tempo", "loudness"]


@app.get("/")
def read_root():
    message = ("Welcome to the Music Recommender API! "
               "Use /recommend/{playlist_id} to get song recommendations.")
    return {"message": message}


@app.post("/trackdata")
def get_track_data(data: PlaylistData):
    try:
        username = data.username
        liked_playlist_id = data.playlist_url.split("/")[-1].split("?")[0]

        liked_tracks = fetch_playlist_features(liked_playlist_id)
        if not liked_tracks:
            raise HTTPException(status_code=404,
                                detail="Playlist not found or empty")

        liked_data = pd.DataFrame(liked_tracks)
        liked_data, _ = normalize_features(liked_data)
        save_to_db(username, liked_playlist_id, liked_data)

        return {"status": "success", "playlist_id": liked_playlist_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{playlist_id}")
def recommend(playlist_id: str):
    # check if playlist exists
    liked_songs = load_playlist_songs(playlist_id)
    if liked_songs.empty:
        raise HTTPException(status_code=404, detail="Playlist not found")

    candidate_df = load_candidate_songs()

    # convert features to tensors
    X_candidate = torch.tensor(candidate_df[features_columns].values,
                               dtype=torch.float32).to(device)
    X_liked = torch.tensor(liked_songs[features_columns].values,
                           dtype=torch.float32).to(device)

    (playlist_emb, cand_emb, liked_emb, top_idx, rec_songs, scores) = (
        recommend_songs(X_liked, X_candidate, candidate_df))

    recommendations = []
    # show top recommendations with similarity
    for rank, idx in enumerate(top_idx):
        song = candidate_df.iloc[idx.item()]
        recommendations.append({
            "track_name": song["track_name"],
            "artist_name": song["artist"],
            "score": float(scores[rank].item())
        })

    return {"playlist_id": playlist_id, "recommendations": recommendations}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
