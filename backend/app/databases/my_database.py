import sqlite3
import os

# set up db
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "databases/my_database.db")

try:
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()

        cursor.execute("PRAGMA foreign_keys = ON;")

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlists (
                playlist_id TEXT PRIMARY KEY,
                username TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username)
                    ON DELETE CASCADE
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS songs (
                track_id TEXT PRIMARY KEY,
                track_name TEXT,
                artist TEXT,
                parent_genre TEXT,
                cover_url TEXT,
                preview_url TEXT,
                danceability REAL,
                energy REAL,
                valence REAL,
                speechiness REAL,
                acousticness REAL,
                instrumentalness REAL,
                liveness REAL,
                tempo REAL,
                loudness REAL
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlist_songs (
                playlist_id TEXT,
                track_id TEXT,
                PRIMARY KEY (playlist_id, track_id),
                FOREIGN KEY (playlist_id)
                    REFERENCES playlists(playlist_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (track_id) REFERENCES songs(track_id)
            );
        ''')

except sqlite3.Error as e:
    print(f"A database error occurred: {e}")

# The connection is automatically closed upon exiting the 'with' block
