import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import styles from "../styles/button.module.css";

interface Recommendation {
  track_name: string;
  track_id: string;
  artist_name: string;
  cover_art: string;
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
      <div
        className={` mx-auto lg:max-w-310 max-w-150 rounded-t w-full h-10 ${styles.playertop}`}
      ></div>

      {recommendations.length > 0 && (
        <div
          className={`lg:max-w-310 max-w-150 mx-auto lg:flex justify-center gap-5 rounded-b-2xl ${styles.player}`}
        >
          {recommendations.map((rec, idx) => (
            <div
              key={idx}
              className={` p-6 w-100 rounded-lg mx-auto text-center flex justify-center lg:flex-col gap-5 items-center ${styles.player}`}
            >
              <div className={` rounded-full p-5 ${styles.cdholder} `} >
                <img
                  src={rec.cover_art}
                  alt="Album cover"
                  width="200"
                  height="200"
                  className={` rounded-full ${styles.spin_image}`}
                />
              </div>
              <div className=" mx-auto text-center lg:flex gap-5 justify-center">
                <div className="mx-auto text-center p-2">
                  <h2 className="text-xl font-bold">{rec.track_name}</h2>
                  <p className="text-cyan-300 text-lg">{rec.artist_name}</p>
                </div>
                <div className="flex flex-col items-center">
                  {rec.track_id && (
                    <a
                      type="submit"
                      href={`https://open.spotify.com/track/${rec.track_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`${styles.button} ${styles.buttonSm}`}
                    ></a>
                  )}
                  <p> Open on Spotify</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Recommendations;
