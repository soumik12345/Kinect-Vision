import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def segment_skin(frame):
    lower = np.array([0, 40, 30], dtype = "uint8")
    upper = np.array([43, 255, 254], dtype = "uint8")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 1)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    print(skinMask)
    return cv2.bitwise_and(frame, frame, mask = skinMask) 

def segment_skin_2(image):
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = np.uint8)
    upper = np.array([20, 255, 255], dtype = np.uint8)
    skin_mask = cv2.inRange(img, lower, upper)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    skin = cv2.bitwise_and(img, img, mask = skin_mask)
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

def cluster(image):
    red = image[:, :, 0]
    X = image[:, :, 0].flatten()
    X = np.reshape(X, (X.shape[0], 1))
    y_pred = KMeans(n_clusters = 2, random_state = 170).fit_predict(X)
    y_pred = np.invert(y_pred.reshape(image.shape[0], image.shape[1]).astype(np.uint8))
    return y_pred
    # return cv2.bitwise_and(image, image, mask = y_pred)

def red_channel(image):
    img = image[:, :, 0]
    edge = cv2.Canny(image, 100, 200)
    return edge


cap = cv2.VideoCapture('./Test Videos/Srinjoy.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame', segment_skin_2(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# img = cv2.imread('Sign-Language-Digits-Dataset/Dataset/0/IMG_1118.JPG')

# while True:
#     cv2.imshow('Image', cluster(img))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# plt.imshow(cluster(img), cmap = 'gray')
# plt.show()