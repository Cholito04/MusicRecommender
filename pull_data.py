# run pip install spotipy
# run pip install pandas
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import requests


# Authenticate using client credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="1087a50f38b64d5f9264509b58d129c0",
    client_secret="04958a2e376941d1ad96d96712522f53"
))

# API settings for ReccoBeats
RECCO_BEATS_API_URL = "https://api.reccobeats.com/v1/audio-features?ids="
RECCO_BEATS_HEADERS = {
    'Accept': 'application/json'
}


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
        tracks.append({
            'track_id': track['id'],
            'track_name': track['name'],
            'artist_name': track['artists'][0]['name'],
            "danceability": features.get("danceability"),
            "energy": features.get("energy"),
            "valence": features.get("valence"),
            "speechiness": features.get("speechiness"),
        })
    return tracks


def get_candidate_tracks(c_playlist_id, filename="candidate_tracks.csv"):
    # Check if CandidateTracks csv exists

    if pd.read_csv(filename) is not None:
        print("Candidate tracks loaded from CSV.")
        return pd.read_csv(filename)
    else:
        print("Candidate tracks CSV not found creating new DataFrame.")
        candidate_tracks = fetch_playlist_features(c_playlist_id)

        # Create DataFrame
        candidate_tracks = pd.DataFrame(candidate_tracks)

        # save to csv
        candidate_tracks.to_csv("candidate_tracks.csv", index=False)
        return candidate_tracks


def main():
    # ask user for public playlist link
    # just press enter to use default playlist
    liked_playlist_id = input("Enter the public playlist share link: ")
    if liked_playlist_id == "":
        liked_playlist_id = "https://open.spotify.com/playlist/10MGklkCciIqPR9cHRbQJW?si=9bdc9d45f4cb400f"
    liked_tracks = fetch_playlist_features(liked_playlist_id)

    # Create DataFrame
    liked_data = pd.DataFrame(liked_tracks)
    print(liked_data, "\n")

    # save to csv
    liked_data.to_csv("liked_songs.csv", index=False)
    c_playlist_id = "7akjhHXrGWcbFZeRjfZzp2"  # Example candidate playlist ID

    candidate_tracks = get_candidate_tracks(c_playlist_id)
    print(candidate_tracks.head)


if __name__ == "__main__":
    main()
