import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";

interface Recommendation {
  track_name: string;
  artist_name: string;
  score: number;
}

function Recommendations() {
  const [error, setError] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const location = useLocation();
  const navigate = useNavigate();

  const playlist_id = location.state?.playlist_id;

  useEffect(() => {
    if (!playlist_id) {
      setError("No playlist ID found. Please enter a playlist first.");
      navigate("/playlistenter");
      return;
    }

    async function fetchRecommendations() {
      try {
        const { data } = await axios.get(
          `http://127.0.0.1:8000/recommend/${playlist_id}`,
        );
        setRecommendations(data.recommendations);
      } catch (err: any) {
        if (err.response?.status === 404) {
          setError("Playlist not found");
        } else {
          setError("Failed to load recommendations");
        }
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    fetchRecommendations();
  }, [playlist_id, navigate]);

  if (loading) {
    return (
      <div className="text-white text-center py-36">
        <h1 className="text-4xl">Loading recommendations...</h1>
      </div>
    );
  }

  return (
    <div className="text-white w-full px-10 py-36 mx-auto">
      <h1 className="text-6xl text-center mb-10">Your Recommendations</h1>

      {error && <p className="text-red-400 text-center text-xl">{error}</p>}

      {recommendations.length > 0 && (
        <div className="max-w-4xl mx-auto">
          {recommendations.map((rec, idx) => (
            <div key={idx} className="bg-gray-800 p-6 rounded-lg mb-4">
              <h2 className="text-2xl font-bold">{rec.track_name}</h2>
              <p className="text-gray-400 text-lg">{rec.artist_name}</p>
              <p className="text-green-400 mt-2">
                Match: {(rec.score * 100).toFixed(1)}%
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Recommendations;
