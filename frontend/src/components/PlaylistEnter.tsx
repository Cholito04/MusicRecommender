import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import axios from "axios";

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
      const { data } = await axios.post(`http://127.0.0.1:8000/trackdata`, {
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
        <div className="mx-auto lg:max-w-310 max-w-200 p-3">
          <div className="flex justify-center gap-2">
            <input
              name="playlist"
              placeholder="Paste entire URL"
              className="text-white max-w-200 p-2 rounded mb-6 text-2xl bg-gray-800"
              required
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading}
              className="text-white max-h-12 px-4 rounded text-xl bg-indigo-800 hover:bg-indigo-500 transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Processing..." : "Generate"}
            </button>
          </div>

          {/* Loading indicator */}
          {loading && (
            <div className="mt-4">
              <p className="text-xl text-gray-300 mb-2">
                Analyzing your playlist...
              </p>
              <div className="w-64 h-2 bg-gray-700 rounded-full overflow-hidden mx-auto">
                <div className="h-full bg-linear-to-r from-indigo-500 to-purple-500 animate-pulse"></div>
              </div>
            </div>
          )}

          {error && <p className="text-red-400 text-lg mt-4">{error}</p>}
        </div>
      </div>
    </form>
  );
}

export default PlaylistEnter;
