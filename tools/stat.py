import sys
sys.path.insert(0,".")
import os
import pymatting
#$import matting.utils.config as config
import numpy as np
import cv2

def main():

    #train_fg_path = config.fg_path
    #train_alpha_path = config.alpha_path
    train_fg_path = "/mnt/d/DataSets/Adobe_Deep_Matting_Dataset/all_fg"
    train_alpha_path = "/mnt/d/DataSets/Adobe_Deep_Matting_Dataset/all_alpha"

    my_sum = 0
    for f in os.listdir(train_fg_path):
        if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
            img = cv2.imread(os.path.join(train_fg_path, f))
            alpha = cv2.imread(os.path.join(train_alpha_path, f), 0)
            my_sum += np.sum((alpha < 255) & (alpha > 0))
            print(my_sum)


if __name__ == "__main__":
    main()
