//import ListGroup from "./components/ListGroup";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Hero from "./components/Hero";
import PlaylistEnter from "./components/PlaylistEnter";
import UserEnter from "./components/UserEnter";
import Getuser from "./components/Getuser";
import Recommendations from "./components/Recomendations";

function App() {
  return (
    <div>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Hero />} />
          <Route path="/playlistenter" element={<PlaylistEnter />} />
          <Route path="/userenter" element={<UserEnter />} />
          <Route path="/getuser" element={<Getuser />} />
          <Route path="/reccomendations" element={<Recommendations />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
