# run pip install spotipy
# run pip install pandas
import pandas as pd
import os
import sqlite3
# pip install scikit-learn
# used to normalize data
from pull_data import fetch_playlist_features, normalize_features
# set up db
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "databases/my_database.db")


def get_candidate_tracks(c_playlist_id, filename="candidate_tracks.csv"):
    # Check if CandidateTracks csv exists

    if os.path.exists(filename):
        print("Candidate tracks loaded from CSV.")
        return pd.read_csv(filename)
    else:
        print("Candidate tracks CSV not found creating new DataFrame.")
        candidate_tracks = fetch_playlist_features(c_playlist_id)

        # create DataFrame
        candidate_tracks = pd.DataFrame(candidate_tracks)

        # normalize the data
        candidate_tracks, scaler = normalize_features(candidate_tracks)

        return candidate_tracks


def save_canidate_to_db(candidate_tracks):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        # insert each song and link it to the playlist if song not in playlist
        for _, row in candidate_tracks.iterrows():
            cursor.execute('''
                INSERT OR IGNORE INTO songs (
                    track_id, track_name, artist, parent_genre, cover_url,
                    danceability, energy, valence, speechiness,
                    acousticness, instrumentalness, liveness, tempo, loudness
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['track_id'], row['track_name'], row['artist_name'],
                str(row['genre']), row['cover_url'],
                row['danceability'], row['energy'], row['valence'],
                row['speechiness'], row['acousticness'],
                row['instrumentalness'], row['liveness'],
                row['tempo'], row['loudness']
            ))
        conn.commit()


def main():

    c_playlist_id = "7akjhHXrGWcbFZeRjfZzp2"  # Example candidate playlist ID

    candidate_tracks = get_candidate_tracks(c_playlist_id)
    save_canidate_to_db(candidate_tracks)
    print(candidate_tracks.head())


if __name__ == "__main__":
    main()
