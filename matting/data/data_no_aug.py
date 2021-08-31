import torch
import cv2
import os
import random
import numpy as np
from torchvision import transforms
import logging

from PIL import Image

from ..utils import config

interpolation_list = [cv2.INTER_CUBIC]

def random_resize(input, size):
    return cv2.resize(input, size, interpolation = cv2.INTER_CUBIC)

def random_warpAffine(input, rot_mat, size):
    return cv2.warpAffine(input, rot_mat, size, flags=cv2.INTER_CUBIC)

def random_flip(alpha, fg, bg):
    if random.random() < 0.5:
        fg = cv2.flip(fg, 1)
        alpha = cv2.flip(alpha, 1)
    if random.random() < 0.5:
        bg = cv2.flip(bg, 1)
    return alpha, fg, bg

def get_files(mydir):
    res = []
    for f in os.listdir(mydir):
        if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
            res.append(os.path.join(mydir, f))
    return res

class MatDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.logger = logging.getLogger("DeepImageMatting")

        self.args = args

        self.cnt = len(config.fine_tuning_img_list)
        assert(self.cnt > 0)

        for name in config.fine_tuning_img_list:
            assert(os.path.exists(name))
        for name in config.fine_tuning_alpha_list:
            assert(os.path.exists(name))
        for name in config.fine_tuning_trimap1_list:
            assert(os.path.exists(name))
        for name in config.fine_tuning_trimap2_list:
            assert(os.path.exists(name))

        self.logger.info("Matting Dataset image number: {}".format(self.cnt))

        self.ToTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean = config.mean, std = config.std)

    def __len__(self):
        return len(config.fine_tuning_img_list)

    def __getitem__(self, index):
        image_path = config.fine_tuning_img_list[index]
        alpha_path = config.fine_tuning_alpha_list[index]
        if random.random() < 0.5:
            trimap_path = config.fine_tuning_trimap1_list[index]
        else:
            trimap_path = config.fine_tuning_trimap2_list[index]


        img_info = [image_path, alpha_path, trimap_path]

        # read fg, alpha
        img = cv2.imread(image_path)[:, :, :3]
        alpha = cv2.imread(alpha_path, 0)
        trimap = cv2.imread(trimapa_path, 0)

        img_info.append(img.shape)

        h, w = img.shape[0], img.shape[1]
        # pad to meet h % 64 ==0 and w % 64 == 0
        theDivider = 64
        if h % theDivider == 0:
            pad_h = 0
        else:
            pad_h = theDivider - h % theDivider

        if w % theDivider == 0:
            pad_w = 0
        else:
            pad_w = theDivider - w % theDivider
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0,0)), mode = "reflect")
        alpha = np.pad(alpha, ((0, pad_h), (0, pad_w), (0,0)), mode = "reflect")
        trimap = np.pad(trimap, ((0, pad_h), (0, pad_w)), mode = "reflect")

        h, w = img.shape[0], img.shape[1]

        if random.random() < 0.2:
            img = 255 - img
        if random.random() < 0.2:
            img = img[:,:,np.random.permutation(3)]

        ## debug
        cv2.imwrite("result/debug/debug_{}_img.png".format(index), img)
        cv2.imwrite("result/debug/debug_{}_alpha.png".format(index), alpha)
        cv2.imwrite("result/debug/debug_{}_trimap.png".format(index), trimap)

        #########

        ###################################
        # blur
        #if random.random() < 0.3:
        #    t1 = random.randint(0,2)
        #    t2 = random.randint(0,2)
        #    fg = cv2.blur(fg, (2*t1 + 1, 2*t2+1))
        #    alpha = cv2.blur(alpha, (2*t1 + 1, 2*t2+1))
        # sharpen
        #if random.random() < 0.3:
        #    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        #    fg = cv2.filter2D(fg, -1, kernel)
        #    alpha = cv2.filter2D(alpha, -1, kernel)
        ##################################

        img_rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_norm = self.ToTensor(img_rgb)

        ##################################
        # blur
        #if random.random() < 0.3:
        #    t1 = random.randint(0,2)
        #    t2 = random.randint(0,2)
        #    bg = cv2.blur(bg, (2*t1 + 1, 2* t2 + 1))
        # sharpen
        #if random.random() < 0.3:
        #    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        #    bg = cv2.filter2D(bg, -1, kernel)
        ##################################

        alpha = torch.from_numpy(alpha.astype(np.float32)[np.newaxis, :, :])
        trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :])

        img_norm = self.normalize(img_norm)

        return img_norm, alpha, trimap, img_info

