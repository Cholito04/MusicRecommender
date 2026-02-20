from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sqlite3
import os

router = APIRouter()


# Pydantic model for request body
class User(BaseModel):
    username: str


# Path to your DB
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "databases/my_database.db")


@router.post("/createusers")
def create_user(user: User):
    print(f"DB_PATH: {DB_PATH}")  # Debug line
    print(f"File exists: {os.path.exists(DB_PATH)}")  # Debug line
    """
    Create a new user.
    If the username already exists, returns 400.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.execute(
                "INSERT INTO users (username) VALUES (?)",
                (user.username,)
            )
            conn.commit()

        return {"username": user.username, "status": "created"}

    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{username}")
def get_user(username: str):
    """Get __ user"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        SELECT username
        FROM users
        WHERE username = ?""",
                       (username,))
        row = cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="User not found")
        conn.close()
        return {"user": username}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user}/playlists")
def get_userPlaylist(username: str):
    """Get __ user playlists"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        SELECT playlist_id
        FROM playlists
        WHERE username = ?""",
                       (username,))
        playlist = [{"playlist_id": row[0]} for row in cursor.fetchall()]
        if playlist is None:
            raise HTTPException(status_code=404, detail="User has no playlist")
        conn.close()
        return {"playlist": playlist}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
