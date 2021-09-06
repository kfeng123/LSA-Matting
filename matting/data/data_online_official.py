import torch
import cv2
import os
import random
import numpy as np
from torchvision import transforms
import logging

from PIL import Image

from ..utils import config

#interpolation_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]
interpolation_list = [cv2.INTER_CUBIC]

def random_resize(input, size):
    tmp = random.randint(0, len(interpolation_list)-1)
    return cv2.resize(input, size, interpolation = interpolation_list[tmp])

def random_warpAffine(input, rot_mat, size):
    tmp = random.randint(0, len(interpolation_list)-1)
    return cv2.warpAffine(input, rot_mat, size, flags=interpolation_list[tmp])

def original_trimap(alpha):
    out = {}

    fg = np.equal(alpha, 255).astype(np.uint8)
    fg_wide = np.not_equal(alpha, 0).astype(np.uint8)
    unknown = fg_wide - fg

    distanceTrans = cv2.distanceTransform(1-unknown, cv2.DIST_L2, 0)
    unknown =  np.logical_or(np.logical_and(distanceTrans <= np.random.randint(1, 25), fg < 1),\
                np.logical_and(distanceTrans <= np.random.randint(1, 25), fg > 0) )
    trimap = fg * 255
    trimap[unknown] = 128

    out['trimap'] = trimap.astype(np.uint8)

    return out

class random_rotation:
    def __init__(self):
        self.degrees = [-180, 180]
        self.center_h_range = [-10, 10]
        self.center_w_range = [-10, 10]

    def __call__(self, alpha, fg, bg):
        h = fg.shape[0]
        w = fg.shape[1]

        # alpha and fg
        centerh = h//2 + random.randint(self.center_h_range[0], self.center_h_range[1])
        centerw = w//2 + random.randint(self.center_w_range[0], self.center_w_range[1])
        fg_degree = random.randint(self.degrees[0], self.degrees[1])
        rot_mat = cv2.getRotationMatrix2D((centerw, centerh), fg_degree, 1)

        alpha = random_warpAffine(alpha, rot_mat, (w, h))
        fg = random_warpAffine(fg, rot_mat, (w, h))

        # bg
        bg_centerh = h//2 + random.randint(self.center_h_range[0], self.center_h_range[1])
        bg_centerw = w//2 + random.randint(self.center_w_range[0], self.center_w_range[1])
        bg_degree = random.randint(self.degrees[0], self.degrees[1])
        bg_rot_mat = cv2.getRotationMatrix2D((bg_centerw, bg_centerh), bg_degree, 1)
        bg = random_warpAffine(bg, bg_rot_mat, (w, h))

        return alpha, fg, bg

# new gamma augmentation
def gamma_aug(img, gamma):
    if random.random() < 0.5:
        return ( 255. * pow(img/255., gamma) ).clip(0, 255).astype(np.uint8)
    else:
        return ( 255. * ( 1-pow(1-img/255., gamma) ) ).clip(0, 255).astype(np.uint8)

def random_crop(alpha, fg, bg, crop_h, crop_w):
    h, w = alpha.shape
    unknown = (alpha > 0) & (alpha < 255)
    target = np.where(alpha > 0)
    delta_h = int(crop_h/2)
    delta_w = int(crop_w/2)
    center_h = int(h/2)
    center_w = int(w/2)
    if len(target[0]) > 0:
        tmp_counter = 0
        unknown_ratio = unknown.sum() / (h * w)
        while True:
            tmp_counter += 1
            rand_ind = np.random.randint(len(target[0]))
            center_h = target[0][rand_ind]
            if center_h < delta_h:
                center_h = delta_h
            if center_h > h - delta_h:
                center_h = h - delta_h
            center_w = target[1][rand_ind]
            if center_w < delta_w:
                center_w = delta_w
            if center_w > w - delta_w:
                center_w = w - delta_w

            start_h = center_h - delta_h
            start_w = center_w - delta_w
            end_h   = center_h + delta_h
            end_w   = center_w + delta_w
            tmp = unknown[start_h : end_h, start_w : end_w].sum() / (4 * delta_h * delta_w)
            if  tmp >= unknown_ratio * 0.8:
                break
            if tmp_counter >= 5:
                break

    start_h = center_h - delta_h
    start_w = center_w - delta_w
    end_h   = center_h + delta_h
    end_w   = center_w + delta_w

    alpha  =alpha [start_h : end_h, start_w : end_w]
    fg    = fg   [start_h : end_h, start_w : end_w]
    bg    = bg   [start_h : end_h, start_w : end_w]
    return alpha, fg, bg

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
        self.train_size_h = config.train_size_h
        self.train_size_w = config.train_size_w
        self.crop_size_h = config.crop_size_h
        self.crop_size_w = config.crop_size_w
        assert(len(self.crop_size_h) == len(self.crop_size_w))

        self.total_fg_list = config.total_fg_list
        self.total_alpha_list = config.total_alpha_list
        self.total_bg_list = config.total_bg_list

        self.cnt = len(self.total_fg_list)
        assert(self.cnt > 0)

        for name in self.total_fg_list:
            assert(os.path.exists(name))
        for name in self.total_alpha_list:
            assert(os.path.exists(name))
        for name in self.total_bg_list:
            assert(os.path.exists(name))

        self.logger.info("Matting Dataset foreground number: {}".format(self.cnt))

        self.random_rotation = random_rotation()
        self.ColorJitter = transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2)
        self.ToTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean = config.mean, std = config.std)

    def __len__(self):
        return len(self.total_bg_list)

    def __getitem__(self, index):
        image_bg_path = self.total_bg_list[index]
        fg_index = random.randint(0, len(self.total_fg_list) - 1 )
        image_alpha_path = self.total_alpha_list[fg_index]
        image_fg_path = self.total_fg_list[fg_index]

        img_info = [image_fg_path, image_alpha_path, image_bg_path]

        # read fg, alpha
        alpha = cv2.imread(image_alpha_path, 0)
        fg = cv2.imread(image_fg_path)[:, :, :3]
        bg = cv2.imread(image_bg_path)[:, :, :3]

        bh, bw, bc, = fg.shape
        if self.train_size_h > bh:
            tmp = self.train_size_h - bh
            fg = cv2.copyMakeBorder(fg, tmp, tmp, tmp, tmp, cv2.BORDER_DEFAULT)
            alpha = cv2.copyMakeBorder(alpha, tmp, tmp, tmp, tmp, cv2.BORDER_DEFAULT)
            bh, bw, bc, = fg.shape
        if self.train_size_w > bw:
            tmp = self.train_size_w - bw
            fg = cv2.copyMakeBorder(fg, tmp, tmp, tmp, tmp, cv2.BORDER_DEFAULT)
            alpha = cv2.copyMakeBorder(alpha, tmp, tmp, tmp, tmp, cv2.BORDER_DEFAULT)
            bh, bw, bc, = fg.shape

        img_info.append(fg.shape)

        bg = random_resize(bg, (bw, bh))

        #### random rotation
        if random.random() < 0.5:
            alpha, fg, bg = self.random_rotation(alpha, fg, bg)

        rand_ind = random.randint(0, len(self.crop_size_h) - 1)
        cur_crop_h = self.crop_size_h[rand_ind]
        cur_crop_w = self.crop_size_w[rand_ind]

        wratio = float(cur_crop_w) / bw
        hratio = float(cur_crop_h) / bh
        ratio = max(wratio, hratio)
        if ratio > 1:
            nbw = int(bw * ratio + 1.0)
            nbh = int(bh * ratio + 1.0)
            alpha = random_resize(alpha, (nbw, nbh))
            fg = random_resize(fg, (nbw, nbh))
            bg = random_resize(bg, (nbw, nbh))

        alpha, fg, bg = random_crop(alpha, fg, bg, cur_crop_h, cur_crop_w)
        alpha, fg, bg = random_flip(alpha, fg, bg)

        if self.train_size_h != fg.shape[0] or self.train_size_w != fg.shape[1]:
            alpha = random_resize(alpha,  (self.train_size_w, self.train_size_h))
            fg    =random_resize(fg,     (self.train_size_w, self.train_size_h))
            bg    =random_resize(bg,     (self.train_size_w, self.train_size_h))

        if random.random() < 0.3:
            # gamma is from 0.5 to 2
            if random.random() < 0.5:
                alpha = gamma_aug(alpha, random.random()/2 + 0.5)
            else:
                alpha = gamma_aug(alpha, random.random() + 1.)

        #if random.random() < 0.3:
        #    tmp = [random.random() * 255 for i in range(3)]
        #    for i in range(3):
        #        weight = random.random()
        #        fg[:,:,i] = fg[:,:,i] * weight + tmp[i] * (1-weight)

        if random.random() < 0.3:
            fg = 255 - fg
        if random.random() < 0.3:
            fg = fg[:,:,np.random.permutation(3)]

        trimap_dict = original_trimap(alpha)
        trimap = trimap_dict['trimap']
        ## debug
        #cv2.imwrite("result/debug/debug_{}_fg.png".format(index),fg)
        #cv2.imwrite("result/debug/debug_{}_bg.png".format(index),bg)
        #cv2.imwrite("result/debug/debug_{}_trimap.png".format(index),trimap)

        #########

        ###################################
        # blur
        if random.random() < 0.3:
            t1 = random.randint(1,3)
            t2 = random.randint(1,3)
            fg = cv2.GaussianBlur(fg, (2*t1 + 1, 2*t2+1), 0)
            bg = cv2.GaussianBlur(bg, (2*t1 + 1, 2*t2+1), 0)
        #if random.random() < 0.3:
        #    t1 = random.randint(1,3)
        #    t2 = random.randint(1,3)
        #    if random.random() < 0.5:
        #        alpha = cv2.blur(alpha, (2*t1 + 1, 2*t2+1))
        #    else:
        #        alpha = cv2.GaussianBlur(alpha, (2*t1 + 1, 2*t2+1), 0)
        # sharpen
        #if random.random() < 0.0:
        #    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        #    fg = cv2.filter2D(fg, -1, kernel)
        #    bg = cv2.filter2D(bg, -1, kernel)
        ##################################

        # resize
        if random.random() < 0.3:
            h, w = fg.shape[0], fg.shape[1]
            if random.random() < 0.5:
                fg = cv2.resize(fg, (960, 960), interpolation = cv2.INTER_CUBIC)
            else:
                fg = cv2.resize(fg, (320, 320), interpolation = cv2.INTER_CUBIC)
            fg = cv2.resize(fg, (w, h), interpolation = cv2.INTER_CUBIC)
            if random.random() < 0.5:
                bg = cv2.resize(bg, (960, 960), interpolation = cv2.INTER_CUBIC)
            else:
                bg = cv2.resize(bg, (320, 320), interpolation = cv2.INTER_CUBIC)
            bg = cv2.resize(bg, (w, h), interpolation = cv2.INTER_CUBIC)

        fg_rgb = Image.fromarray(cv2.cvtColor(fg, cv2.COLOR_BGR2RGB))
        fg_rgb = self.ColorJitter(fg_rgb)
        fg_norm = self.ToTensor(fg_rgb)

        bg_rgb = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        bg_rgb = self.ColorJitter(bg_rgb)
        bg_norm = self.ToTensor(bg_rgb)

        alpha = torch.from_numpy(alpha.astype(np.float32)[np.newaxis, :, :])
        trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :])

        fg_norm = self.normalize(fg_norm)
        bg_norm = self.normalize(bg_norm)


        return fg_norm, bg_norm, alpha, trimap, img_info

