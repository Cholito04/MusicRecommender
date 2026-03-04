import { Link } from "react-router-dom";
import styles from "../styles/button.module.css";

function Hero() {
  return (
    <div className="text-cyan-200 w-full lg:px-4 px-10 py-36 mx-auto text-center">
      <div className=" mx-auto lg:max-w-310 max-w-200 p-3 text-center">
        <div
          className={` mx-auto lg:max-w-180 max-w-120 rounded-t-xl w-full h-15 ${styles.playertop}`}
        ></div>
        <div className="flex justify-center flex-col w-full gap-4">
          <div
            className={` md:p-18 p-10 mx-auto lg:max-w-180 max-w-120 rounded-b-3xl w-full ${styles.tv}`}
          >
            <div
              className={`md:p-10 p-4 mx-auto rounded-3xl w-full ${styles.lcd}`}
            >
              <h1 className=" m-2 xl:text-6xl text-2xl lg:max-w-310 max-w-200">
                {" "}
                Music Reccomender
              </h1>
              <p className="lg:text-xl text-sm">
                {" "}
                Recommend songs based of your public playlist or any public
                playlist.
              </p>
            </div>
          </div>
          <div className="text-center m-10">
            <div
              className={`p-10 mx-auto rounded-3xl lg:max-w-90 max-w-80 max-h-40 w-full px-3 flex justify-center lg:gap-24 gap-8 ${styles.signIn}`}
            >
              <div>
                {" "}
                <Link
                  className={` h-[80-px] ${styles.button2}`}
                  to="/getuser"
                ></Link>
                <p className="text-white pr-4">Log in</p>
              </div>
              <div>
                {" "}
                <Link
                  className={` h-[80-px] ${styles.button2}`}
                  to="/userenter"
                ></Link>
                <p className="text-white pr-4">Sign up</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Hero;
