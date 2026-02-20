import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

function Getuser() {
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function handleSubmit(e: React.SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null); // Clear previous errors

    const formData = new FormData(e.currentTarget);
    const username = formData.get("username") as string;

    if (!username) return;

    try {
      const { data } = await axios.get(
        `http://127.0.0.1:8000/user/${username}`,
      );
      console.log("User:", data);
      localStorage.setItem("username", data.username);
      navigate("/playlistenter", { state: { username: data.username } });
    } catch (err: any) {
      setError("Username doesnt exist(case senseitive)");
      console.error(err);
    }
  }
  return (
    <form onSubmit={handleSubmit}>
      <div className="text-white w-full lg:px-4 px-10 py-36 mx-auto text-center">
        <div className="mx-auto py-10">
          <h1 className="text-6xl">Log In</h1>
        </div>
        <div className="mx-auto lg:max-w-310 max-w-200 p-3 gap-2">
          <div className="flex justify-center">
            <input
              name="username"
              placeholder="Enter username"
              className="text-white max-w-200 p-2 rounded w-full mb-6 text-2xl"
              required
            />
            <button
              type="submit"
              className=" text-white max-h-12 mx-auto px-4 rounded text-xl bg-indigo-800
            hover:bg-indigo-500 transition"
            >
              Log In
            </button>
          </div>
        </div>
        {error && <p className="text-red-400 text-lg mb-4">{error}</p>}
      </div>
    </form>
  );
}

export default Getuser;
