import cv2
import os

data_directory_path = "data/train2017"
rgb_data_directory_path = "linear_sample_data/rgb/"
bw_data_directory_path = "linear_sample_data/bw/"
counter = 0


for file in os.listdir(data_directory_path):
    # READ THE IMAGE
    rgb_image = cv2.imread(data_directory_path +"/"+ file)

    # CONVERT TO BLACK & WHITE
    bw_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # SAVE BOTH IMAGES IN THEIR FOLDER RESPECTIVLY
    cv2.imwrite(filename=rgb_data_directory_path + "image_" + str(counter) + ".jpg", img=rgb_image)
    cv2.imwrite(filename=bw_data_directory_path + "image_" + str(counter) + ".jpg", img=bw_image)
    counter += 1

    print(file)



