# Kinect-Vision

A computer vision based gesture detection system that automatically detects the number of fingers as a hand gesture and enables you to control simple button pressing games using you hand gestures. Currently the system has been tested on the [**T-Rex Runner** game](http://www.trex-game.skipser.com/).

<img src="https://github.com/soumik12345/Kinect-Vision/blob/master/2019-03-30%20(1).png">

## Installation

1. Clone the repo using `git clone https://github.com/soumik12345/Kinect-Vision`
2. Use `cd Kinect-Vision` to get inside the folder
3. Create a new conda environment using `conda create --name kinect_vision`
4. Activate the environment using `activate kinect_vision`
5. Install the requirements using `pip install -r requirements.txt`

## Run The Program

1. Run the program using `python3 main.py` or `python main.py` or `ipython main.py`
2. Select the Camera port (choose `0` if you are using a laptop)
3. Tune the upper and lower thresholds using the trackbars unless the gestures are being detected accurately enough. Ideally the lower threshold is around `130` and the upper threshold is `255`.
4. Once the detection is working satisfactorily, switch on `Game On` and open the game window
5. If you want to change the control scheme, you can do so by editing the `config.json` file.

## Demo

<img src="https://github.com/soumik12345/Kinect-Vision/blob/master/output.gif">
