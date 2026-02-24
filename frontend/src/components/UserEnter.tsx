import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import styles from "../styles/button.module.css";

function UserEnter() {
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function handleSubmit(e: React.SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);

    const formData = new FormData(e.currentTarget);
    const username = formData.get("username") as string;

    if (!username) return;

    try {
      const { data } = await axios.post("http://127.0.0.1:8000/createusers", {
        username,
      });

      console.log("User created:", data);
      localStorage.setItem("username", data.username);
      navigate("/playlistenter", { state: { username: data.username } });
    } catch (err: any) {
      if (err.response?.status === 400) {
        setError("Username already exists");
      } else {
        setError("Network error. Please check your connection.");
      }
      console.error(err);
    }
  }
  return (
    <form onSubmit={handleSubmit}>
      <div className="text-white lg:max-w-210 w-full lg:px-4 px-10 py-36 mx-auto text-center">
        <div className="mx-auto py-10 ">
          <h1 className="text-6xl">Create Username</h1>
        </div>
        <div
          className={` mx-auto lg:max-w-210 max-w-150 rounded-t-lg w-full lg:h-20 h-12 ${styles.playertop}`}
        ></div>
        <div
          className={` mx-auto lg:max-w-210 max-w-150 w-full rounded-b-2xl shadow-xl py-5 ${styles.player}`}
        >
          <div className="mx-auto lg:max-w-210 max-w-100 p-3">
            <div className="flex justify-center gap-10 mx-auto ">
              <input
                name="username"
                placeholder="Enter username"
                className={`text-black lg:max-w-100 md:w-80 sm:w-60 p-2 rounded-xl mb-6 text-2xl ${styles.lcd}`}
                required
              />
              <div>
                <button type="submit" className={styles.button}></button>
                <p className="text-gray-100">Enter</p>
              </div>
            </div>
          </div>
        </div>
        {error && <p className="text-red-500 text-lg mx-auto p-4">{error}</p>}
      </div>
    </form>
  );
}

export default UserEnter;
