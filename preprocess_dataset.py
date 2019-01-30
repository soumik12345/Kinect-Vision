import os, cv2
from tqdm import tqdm
from preprocess import red_channel

DATASET_LOCATION = './Sign-Language-Digits-Dataset/Dataset/'
CLASSES = sorted(os.listdir(DATASET_LOCATION))
PREPROCESSED_DATASET_LOCATION = './Sign-Language-Digits-Dataset/Preprocessed-Dataset/'

def preprocess_dataset():
    try:
        os.mkdir(PREPROCESSED_DATASET_LOCATION)
        for _class in CLASSES:
            os.mkdir(PREPROCESSED_DATASET_LOCATION + _class)
    except:
        pass
    
    for _class in CLASSES:
        data_loc = DATASET_LOCATION + _class + '/'
        image_files = os.listdir(data_loc)
        c = 1
        for _file in tqdm(image_files):
            image = red_channel(cv2.imread(data_loc + _file))
            cv2.imwrite(PREPROCESSED_DATASET_LOCATION + _class + '/' + str(c) + '.jpg', image)
            c += 1

preprocess_dataset()