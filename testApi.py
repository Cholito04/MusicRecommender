#run pip install spotipy
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Authenticate using client credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="1087a50f38b64d5f9264509b58d129c0",
    client_secret="04958a2e376941d1ad96d96712522f53"
))

# use a public playlist as a user example of liked songs
playlist_id = "10MGklkCciIqPR9cHRbQJW" #id of a public playlist
results = sp.playlist_items(playlist_id)

#collect track IDs and metadata
tracks = []
for item in results['items']:
    track = item['track']
    tracks.append({
        'track_id' : track['id'],
        'track_name': track['name'],
        'artist_name': track['artists'][0]['name']
    })

likedTracks = pd.DataFrame(tracks)
print(likedTracks)

# use a public playlist as a user example of not liked songs
playlist_id = "4elfqzIz3IZHmICnOkjFXM" #id of a public playlist
results = sp.playlist_items(playlist_id)

#collect track IDs and metadata
tracks = []
for item in results['items']:
    track = item['track']
    tracks.append({
        'track_id' : track['id'],
        'track_name': track['name'],
        'artist_name': track['artists'][0]['name']
    })

notLikedTracks = pd.DataFrame(tracks)
print(notLikedTracks)

#combine liked and not liked tracks into a single mixed dataframe 