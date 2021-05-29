import torch
import cv2
import os
import random
import numpy as np
from torchvision import transforms
import logging
from utils import config

alpha_path = "/sdd/WangRui/dataSets/Combined_Dataset/my_clean/train/alpha"
out_path = "/sdd/WangRui/tmp_median_filter"

if not os.path.exists(out_path):
    os.makedirs(out_path)
for name in os.listdir(alpha_path):
    if not name.endswith(".jpg"):
        continue
    alpha = cv2.imread(os.path.join(alpha_path, name), 0)
    cv2.imwrite(os.path.join(out_path, name), alpha )

    trimap_narrow = (alpha == 255).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(out_path,"narrow_" + name), trimap_narrow)
    cv2.imwrite(os.path.join(out_path,"median_narrow_" + name), cv2.medianBlur(trimap_narrow, 3 ))

    trimap_wide = (alpha > 0).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(out_path,"wide_" + name), trimap_wide )
    cv2.imwrite(os.path.join(out_path,"median_wide_" + name), cv2.medianBlur(trimap_wide, 3 ))



