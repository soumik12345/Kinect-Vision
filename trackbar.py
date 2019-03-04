import warnings
warnings.filterwarnings('ignore')
import cv2, math
import numpy as np
from processing_pipeline import process

# global variables
amountOfCameras = 3 # how many cameras you want to use
cameraToUse = 0 #initial camera
cameraChange = True #starts true to connect at start up
camera = cv2.VideoCapture() # empty placeholder

# callback function for the tracker, x is the position value
# you may put whatever name in here
def trackerCallback(x):
    global cameraToUse
    global cameraChange
    if cameraToUse != x:
        print("I change to this camera", x)
        cameraToUse = x
        cameraChange = True

def tracker_callback_nothing(x):
    pass

# function to connect to a camera and replace the videoCapture variable
def connectToCamera():
    global cameraChange
    global camera
    print("Connecting to camera", cameraToUse)
    camera = cv2.VideoCapture(cameraToUse)
    # basic check for connection error
    if camera.isOpened():
        print("Successfully connected")
    else:
        print("Error connecting to camera", cameraToUse)
    cameraChange = False

#initial image with the tracker
img = np.zeros((200,600,3), np.uint8)
cv2.namedWindow('image')

cv2.createTrackbar('Camera', 'image', 0, amountOfCameras - 1, trackerCallback)
cv2.createTrackbar('Lower Threshold', 'image', 0, 255, tracker_callback_nothing)
cv2.createTrackbar('Upper Threshold', 'image', 0, 255, tracker_callback_nothing)

shape = None
while True:
    #check if it has to connect to something else
    if cameraChange:
        connectToCamera()
    # if no problems with the current camera, grab a frame
    if camera.isOpened():
        ret, frame = camera.read()
        if ret:
            thresh_lower = cv2.getTrackbarPos('Lower Threshold', 'image')
            thresh_upper = cv2.getTrackbarPos('Upper Threshold', 'image')
            try:
                roi_1, drawing, thresh1, crop_img, count_defects = process(frame, thresh_lower, thresh_upper)
                img = frame
                cv2.putText(
                    img,
                    str(thresh1.shape),
                    (10, 500),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4, (255, 255, 255),
                    2, cv2.LINE_AA
                )
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