import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMG_SHAPE = (128, 128)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, IMG_SHAPE)
    image = image / 255.0 # normalize 
    return image


if __name__ == '__main__':
    # if bw directory is empty,fill it
    BW_IMAGES_PATH = os.path.join("sample_data", "bw")
    RGB_IMAGES_PATH = os.path.join("sample_data", "rgb")
    if len(os.listdir(BW_IMAGES_PATH)) <= 0:
        for image_name in os.listdir(RGB_IMAGES_PATH):
            image = cv2.imread(os.path.join(RGB_IMAGES_PATH, image_name))
            cv2.imwrite(os.path.join(BW_IMAGES_PATH, image_name), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    for image_name in os.listdir(RGB_IMAGES_PATH):
        plt.cla()
        image = cv2.imread(os.path.join(RGB_IMAGES_PATH, image_name))
        image = preprocess_image(image)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.pause(2)
