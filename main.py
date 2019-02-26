import warnings
warnings.filterwarnings('ignore')
import cv2, os
import numpy as np
import matplotlib.pyplot as plt

def process(image):
    cv2.rectangle(image, (60, 60), (300, 300), (0, 255, 0), 4)
    roi = crop_img = image[70 : 300, 70 : 300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)
    lower_red = np.array([0, 150, 50])
    upper_red = np.array([195, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh1 = cv2.threshold(blurred, 129, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(blurred, 129, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    area_of_contour = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    roi_1 = roi.copy()
    roi_2 = roi.copy()
    cv2.rectangle(roi_1, (x, y), (x + w, y + h), (0, 0, 255), 1)
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(roi_2.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 255, 255), 0)
    return roi_1, drawing

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            res, drawing = process(frame)
            cv2.imshow('result', res)
            cv2.imshow('drawing', drawing)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            break
    cap.release()
    cv2.destroyAllWindows()

main()