import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import axios from "axios";
import styles from "../styles/button.module.css";

const API_URL = import.meta.env.VITE_API_URL;

function PlaylistEnter() {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  // Get username from navigation state
  const username = location.state?.username || localStorage.getItem("username");

  async function handleSubmit(e: React.SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null); // Clear previous errors
    setLoading(true); // Start loading

    const formData = new FormData(e.currentTarget);
    const playlist = formData.get("playlist") as string;

    if (!playlist) {
      setLoading(false);
      return;
    }

    // Check if username exists
    if (!username) {
      setError("Please log in first");
      setLoading(false);
      navigate("/getuser"); // or wherever your login page is
      return;
    }

    try {
      const { data } = await axios.post(`${API_URL}/trackdata`, {
        username: username,
        playlist_url: playlist,
      });

      console.log("Playlist URL:", data);
      navigate("/reccomendations", {
        state: { playlist_id: data.playlist_id, username },
      });
    } catch (err: any) {
      setError("Invalid URL or playlist is private");
      console.error(err);
      setLoading(false); // Stop loading on error
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="text-white w-full lg:px-4 px-10 py-36 mx-auto text-center">
        <div className="mx-auto py-10">
          <h1 className="text-6xl">Enter Spotify Playlist URL</h1>
          {username && (
            <p className="text-gray-400 mt-2">Logged in as: {username}</p>
          )}
        </div>
        <div
          className={` mx-auto lg:max-w-190 max-w-150 rounded-t-lg w-full h-10 ${styles.playertop}`}
        >
          {" "}
        </div>
        <div
          className={`mx-auto lg:max-w-190 max-w-150 rounded-b-lg w-ful py-10 ${styles.player}`}
        >
          <div className="flex justify-center lg:gap-20 gap-10">
            <input
              name="playlist"
              placeholder="Paste entire URL"
              className={`text-black lg:max-w-100 w-70 p-2 rounded-xl mb-6 text-2xl ${styles.lcd}`}
              required
              disabled={loading}
            />
            <div className="flex flex-col justify-center ">
              <button
                type="submit"
                disabled={loading}
                className={` h-[80-px] ${styles.button2}`}
              ></button>
              <p> Generate</p>
            </div>
          </div>
          {/* Loading indicator */}
          <div
            className={`mx-auto lg:w-120 w-100 text-center p-5 rounded mt-4 ${styles.lcdcon}`}
          >
            <p className={`text-xl mb-5 rounded-lg h-7 ${styles.lcd}`}>
              {loading ? "Analyzing your playlist..." : ""}
            </p>
            <div className={styles.loadingBar}>
              {/* only animate when loading */}
              {loading && <div className={styles.loadingFill}></div>}
            </div>
          </div>

          {error && <p className="text-red-400 text-lg mt-4">{error}</p>}
        </div>
      </div>
    </form>
  );
}

export default PlaylistEnter;
