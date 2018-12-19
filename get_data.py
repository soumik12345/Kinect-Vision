import numpy as np
from tqdm import tqdm
import os, cv2, sys, re
from time import sleep
from matplotlib import pyplot as plt

def create_dir_structure(class_name):
    try:
        os.mkdir('Data')
    except:
        pass
    os.chdir('./Data/')
    try:
        os.mkdir(class_name)
    except:
        print('Class already present,\nDo you want more images of the same class [y/n]:')
        if input() == 'y':
            pass
        else:
            sys.exit(0)
    os.chdir('../')

def create_delay(n, message):
    print(message)
    for i in range(n, 0, -1):
        print(i)
        sleep(1)

def grab_frames(n_frames):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 120)
    images = []
    for i in range(n_frames):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp_frame = cv2.rectangle(frame, (734, 130), (1010, 498), (0, 255, 0), 3)
        roi = frame[130 : 498, 734 : 1010]
        cv2.imshow('Display', disp_frame)
        cv2.imshow('ROI', roi)
        images.append(roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return images

def save_images(images, class_name):
    arr = [re.findall(r'\d+', file)[0] for file in os.listdir('./Data/' + class_name + '/')]
    scale = 0 if len(arr) == 0 else max(arr)
    for i, img in tqdm(enumerate(images)):
        cv2.imwrite('./Data/' + class_name + '/img_' + str(i + scale) + '.jpg', img)

def main():
    _, class_name, n_frames = sys.argv
    n_frames = int(n_frames)
    create_dir_structure(class_name)
    create_delay(5, 'Prepare to display the hand')
    save_images(grab_frames(n_frames), class_name)

main()