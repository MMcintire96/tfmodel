import os
from glob import glob

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

folder_path = 'all_faces/*'

classes = []
def read_data():
    for indx, folder in enumerate(glob(folder_path)):
        classes.append(os.path.basename(folder))
        for indx, img in enumerate(glob(folder+'/*')):
            try:
                #re sample the images as 64x64
                img = cv2.resize(cv2.imread(img), (64, 64))
                yield (img, os.path.basename(folder))
            except Exception as e:
                print("Resize failed")
                os.remove(img)

def label_img():
    all_x = []
    all_y = []
    i = 0
    for img_tuple in read_data():
        img_data = img_tuple[0]
        folder_name = img_tuple[1]
        if folder_name == 'happy':
            label = 1
        elif folder_name == 'neutral':
            label = 0
        all_x.append(img_data)
        all_y.append(label)
        i += 1
        print('{}--->{}'.format(folder_name, i))
    # returns x_train, and test for x set, then train_test for y set
    x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=0)
    return (x_train, y_train), (x_test, y_test)


def save_data():
    (x_train, y_train), (x_test, y_test) = label_img()
    np.save('data/x_train.npy', x_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/x_test.npy', x_test)
    np.save('data/y_test.npy', y_test)
    return (x_train, y_train) (x_test, y_test)


def load_data():
    x_train = np.load('data/x_train.npy')
    y_train = np.load('data/y_train.npy')
    x_test = np.load('data/x_test.npy')
    y_test = np.load('data/y_test.npy')
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    save_data()
