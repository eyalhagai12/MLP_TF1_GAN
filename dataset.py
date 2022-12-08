import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from preprocess import preprocess_image

def create_dataset(path, batch_size):
    dataset = []
    file_paths = [os.path.join(path, file_name) for file_name in os.listdir(path)]

    for idx in range(0, len(file_paths), batch_size):
        if idx + batch_size < len(file_paths):
            dataset.append(file_paths[idx:idx + batch_size])

    return dataset

# def extract_from_tensor(image_tensor):
#     return np.array(image_tensor).item().decode()

def convert_batch_to_numpy(batch, bw):
    if bw:
        return np.array([preprocess_image(cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)) for file_name in batch])    
    return np.array([preprocess_image(cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)) for file_name in batch])

if __name__ == "__main__":
    print("GPU available: ", tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    batch_size = 32
    bw_dataset = create_dataset("sample_data/bw", batch_size)
    rgb_dataset = create_dataset("sample_data/rgb", batch_size) 

    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    for x, y in zip(bw_dataset, rgb_dataset):
        x = convert_batch_to_numpy(x, True)
        y = convert_batch_to_numpy(y, False)
        for bw_image, rgb_image in zip(x, y):
            axes[0].clear()
            axes[0].imshow(bw_image, cmap="gray")
            
            axes[1].clear()
            axes[1].imshow(rgb_image)

            print(bw_image.shape)
            print(rgb_image.shape)
            print()

            plt.pause(2)
        
        break

