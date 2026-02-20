import { Link } from "react-router-dom";

function Hero() {
  return (
    <div className="text-white w-full lg:px-4 px-10 py-36 mx-auto text-center">
      <div className=" mx-auto lg:max-w-310 max-w-200 p-3">
        <div className="flex justify-center flex-col w-full gap-4">
          <h1 className=" text-6xl lg:max-w-310 max-w-200">
            {" "}
            Music Reccomender
          </h1>
          <p className="text-xl">
            {" "}
            Recommend songs based of your public playlist or any public
            playlist.
          </p>
          <div className=" p-3 max-w-310 max-h-20 w-full px-3 flex justify-center gap-4">
            <Link
              className="h-[60-px] px-6 py-3 rounded-md bg-indigo-800
            hover:bg-indigo-500 transition"
              to="/getuser"
            >
              Log In
            </Link>
            <Link
              className="h-[60-px] px-6 py-3 rounded-md bg-indigo-800
            hover:bg-indigo-500 transition"
              to="/userenter"
            >
              Sign up
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Hero;
