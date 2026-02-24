# run pip install spotipy
# run pip install pandas
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import requests
import os
import sqlite3
# pip install scikit-learn
# used to normalize data
from sklearn.preprocessing import MinMaxScaler


# Authenticate using client credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("client_id"),
    client_secret=os.getenv("client_secret")
))

# API settings for ReccoBeats
RECCO_BEATS_API_URL = os.getenv("RECCO_BEATS_API_URL")
RECCO_BEATS_HEADERS = {
    'Accept': 'application/json'
}

# set up db
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "databases/my_database.db")


def normalize_features(df):
    feature_cols = [
        "danceability", "energy", "valence", "speechiness",
        "acousticness", "instrumentalness", "liveness",
        "tempo", "loudness"
    ]

    # drop rows with missing values
    df = df.dropna(subset=feature_cols)

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def fetch_playlist_features(playlist_id):

    playlist_id = playlist_id.split("/")[-1]  # extract the playlist ID
    playlist_id = playlist_id.split("?")[0]  # extract the playlist ID
    results = sp.playlist_items(playlist_id)

    # collect track IDs and metadata
    tracks = []
    all_tracks = results['items']

    # get all songs from the playlist bc results ['items'] will only return
    # 100 tracks to add the rest we use 'next'
    while results['next']:
        results = sp.next(results)
        all_tracks.extend(results['items'])

    for item in all_tracks:
        track = item['track']
        print(f"adding track {track['name']}")
        artist_id = track['artists'][0]['id']
        artist = sp.artist(artist_id)
        # request audio features for each track from ReccoBeats API
        response = requests.request("GET",
                                    RECCO_BEATS_API_URL + track['id'],
                                    headers=RECCO_BEATS_HEADERS)

        # Parse features safely
        if response.status_code == 200:
            data = response.json()
        else:
            print(f"Failed to get features for {track['name']}")
            continue

        content = data.get('content', [])

        features = content[0] if content else {}
        if features == {}:
            continue  # skip if no features found
        cover_url = (track['album']['images'][0]['url']
                     if track['album']['images'] else None)
        tracks.append({
            'track_id': track['id'],
            'track_name': track['name'],
            'artist_name': track['artists'][0]['name'],
            'genre': artist['genres'],
            'cover_url': cover_url,
            'preview_url': track['preview_url'],
            # audio features
            "danceability": features.get("danceability"),
            "energy": features.get("energy"),
            "valence": features.get("valence"),
            "speechiness": features.get("speechiness"),
            "acousticness": features.get("acousticness"),
            "instrumentalness": features.get("instrumentalness"),
            "liveness": features.get("liveness"),
            "tempo": features.get("tempo"),
            "loudness": features.get("loudness")
        })
    return tracks


# set up database
def save_to_db(username, playlist_id, tracks_df):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        # insert user if they don't exist yet
        cursor.execute('''
            INSERT OR IGNORE INTO users (username) VALUES (?)
        ''', (username,))

        # insert playlist
        cursor.execute('''
            INSERT OR IGNORE INTO playlists
            (playlist_id, username) VALUES (?, ?)
        ''', (playlist_id, username))

        # insert each song and link it to the playlist
        for _, row in tracks_df.iterrows():
            cursor.execute('''
                INSERT OR IGNORE INTO songs (
                    track_id, track_name, artist, parent_genre, cover_url,
                    preview_url, danceability, energy, valence, speechiness,
                    acousticness, instrumentalness, liveness, tempo, loudness
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['track_id'], row['track_name'], row['artist_name'],
                str(row['genre']), row['cover_url'], row['preview_url'],
                row['danceability'], row['energy'], row['valence'],
                row['speechiness'], row['acousticness'],
                row['instrumentalness'], row['liveness'],
                row['tempo'], row['loudness']
            ))

            cursor.execute('''
                INSERT OR IGNORE INTO playlist_songs (playlist_id, track_id)
                VALUES (?, ?)
            ''', (playlist_id, row['track_id']))

        conn.commit()


def main():
    # ask user for public playlist link
    username = input("Enter a username: ").strip()
    while username == "":
        print("Username cannot be empty.")
        username = input("Enter a username: ").strip()
    # Get the playlist URL from user
    liked_playlist_url = input(
        "Enter the public playlist share link: "
    ).strip()
    while liked_playlist_url == "":
        print("Playlist cannot be empty.")
        username = input("Enter a playlist: ").strip()
    liked_playlist_id = liked_playlist_url.split("/")[-1].split("?")[0]
    liked_tracks = fetch_playlist_features(liked_playlist_id)
    liked_data = pd.DataFrame(liked_tracks)

    liked_data, liked_scaler = normalize_features(liked_data)

    # Extract clean playlist_id for saving
    playlist_id = liked_playlist_url.split("/")[-1].split("?")[0]
    save_to_db(f"{username}",
               playlist_id, liked_data)
    print(liked_data, "\n")


if __name__ == "__main__":
    main()
