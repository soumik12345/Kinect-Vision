import warnings
warnings.filterwarnings('ignore')
import cv2, math, pyautogui
import numpy as np
from processing_pipeline import process
from game_control import *

# global variables
amount_of_cameras = 3 # how many cameras you want to use
camera_to_use = 0 #initial camera
camera_change = True #starts true to connect at start up
camera = cv2.VideoCapture() # empty placeholder

# callback function for the tracker, x is the position value
# you may put whatever name in here
def tracker_callback(x):
    global camera_to_use
    global camera_change
    if camera_to_use != x:
        print("I change to this camera", x)
        camera_to_use = x
        camera_change = True

def tracker_callback_nothing(x):
    pass

# function to connect to a camera and replace the videoCapture variable
def connectToCamera():
    global camera_change
    global camera
    print("Connecting to camera", camera_to_use)
    camera = cv2.VideoCapture(camera_to_use)
    # basic check for connection error
    if camera.isOpened():
        print("Successfully connected")
    else:
        print("Error connecting to camera", camera_to_use)
    camera_change = False

#initial image with the tracker
img = np.zeros((200,600,3), np.uint8)
cv2.namedWindow('image')

cv2.createTrackbar('Camera', 'image', 0, amount_of_cameras - 1, tracker_callback)
cv2.createTrackbar('Lower Threshold', 'image', 0, 255, tracker_callback_nothing)
cv2.createTrackbar('Upper Threshold', 'image', 0, 255, tracker_callback_nothing)
cv2.createTrackbar('Game On', 'image', 0, 1, tracker_callback_nothing)

shape = None
config = get_config_data()

while True:
    #check if it has to connect to something else
    if camera_change:
        connectToCamera()
    # if no problems with the current camera, grab a frame
    if camera.isOpened():
        ret, frame = camera.read()
        if ret:
            thresh_lower = cv2.getTrackbarPos('Lower Threshold', 'image')
            thresh_upper = cv2.getTrackbarPos('Upper Threshold', 'image')
            is_game_on = cv2.getTrackbarPos('Game On', 'image')
            try:
                roi_1, drawing, thresh1, crop_img, count_defects = process(frame, thresh_lower, thresh_upper)
                fingers = count_defects + 1
                img = frame
                cv2.putText(
                    img,
                    str(fingers),
                    (520, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255),
                    2, cv2.LINE_AA
                )
                if is_game_on == 1:
                    pyautogui.press(config[str(fingers)])
            except:
                img = frame
    # displays the frame, in case of none, displays the previous one
    cv2.imshow('image', img)
    # if esc button exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
print(shape)